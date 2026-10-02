# Phase 2 — Fix the correctness bugs verified on `main`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contracts](shared-contracts.md)

**Inputs to read first:**

- [src/neurospatial/layout/engines/graph.py:70-153, 316-339](../../../../src/neurospatial/layout/engines/graph.py) — `GraphLayout.build` and `to_linear`. `to_linear` calls `track_linearization.get_linearized_position` on `_build_params_used["graph_definition"]`.
- `track_linearization/core.py` (installed 2.4.0) at `:1001-1014` and `:1206`. The nearest segment is a *position* in `graph.edges()`, but it is then looked up as the `edge_id` attribute. `make_track_graph` (`utils.py:94-95`) assigns `edge_id` = enumeration index, which is the only numbering under which this lookup is correct.
- [src/neurospatial/environment/factories.py:64-75, 127-186, 740, 915-932, 978-1012](../../../../src/neurospatial/environment/factories.py). These assign `edge_id` in `edge_order` order: the W maze gets `edge_id`s 0,2,1,3,4 in enumeration order. Two comments claim `edge_id` is "not consumed by `to_linear()`".
- [src/neurospatial/encoding/egocentric.py:2248, 2320-2323](../../../../src/neurospatial/encoding/egocentric.py), [encoding/directional.py:2296-2347, 2385-2388](../../../../src/neurospatial/encoding/directional.py), [stats/circular.py:1594, 1686-1692, 1797-1800](../../../../src/neurospatial/stats/circular.py) — the three polar axis setups.
- [src/neurospatial/io/nwb/_behavior.py:122-147, 360-383](../../../../src/neurospatial/io/nwb/_behavior.py), [_pose.py:116-125](../../../../src/neurospatial/io/nwb/_pose.py), [_environment.py:1100-1123, 1126-1179](../../../../src/neurospatial/io/nwb/_environment.py), [_adapters.py:26](../../../../src/neurospatial/io/nwb/_adapters.py) — raw `.data[:]` reads and the unit-string copy.
- [src/neurospatial/simulation/models/place_cells.py:30, 201-207](../../../../src/neurospatial/simulation/models/place_cells.py), [simulation/validation.py:59, 163, 217-220](../../../../src/neurospatial/simulation/validation.py). `env.bin_sizes` is a per-bin *volume* (an area in 2-D), but these lines use it as a length.
- [src/neurospatial/ops/calculus.py](../../../../src/neurospatial/ops/calculus.py) (all 391 lines) and [ops/diffusion.py:1-31, 280-331, 1249-1323, 1386-1422](../../../../src/neurospatial/ops/diffusion.py). Diffusion defines `L = M⁻¹(Deg − W)` with `W_ij = A_ij / d_ij`. `_finite_volume_geometry` returns a copy of `env.connectivity` carrying a face measure `"A"` on every edge, plus the cell volumes `M`. `_graph_fv` also replaces track-junction chord distances with along-track lengths.
- [src/neurospatial/environment/core.py:51, 1049-1118](../../../../src/neurospatial/environment/core.py) — `get_differential_operator`, cached through `versioned_cached_property`.
- [docs/reviews/REPOSITORY_AND_MATHEMATICAL_REVIEW_2026-10-01.md §2.1](../../../../docs/reviews/REPOSITORY_AND_MATHEMATICAL_REVIEW_2026-10-01.md) (committed in Phase 1).
- [src/neurospatial/ops/egocentric.py:655-817, 820-862, 928](../../../../src/neurospatial/ops/egocentric.py) — `heading_from_velocity` (forward-difference velocity at `:768-771`, low-speed mask at `:784`, interpolation call at `:815`) and `_interpolate_heading_circular` (chord interpolation at `:855-860`; also called by `heading_from_body_orientation` at `:928`). Task 6 and Task 10.
- [src/neurospatial/ops/basis.py:575-757, 759-957, 959-1004](../../../../src/neurospatial/ops/basis.py) — `heat_kernel_wavelet_basis` (scale docs `:599-618`, "Laplacian weighting" Notes `:654-657`, `nx.laplacian_matrix(weight="distance")` at `:722-724`), `chebyshev_filter_basis` (`:902-911`) and `_estimate_spectral_radius`. Task 7.
- [src/neurospatial/events/alignment.py:343-500](../../../../src/neurospatial/events/alignment.py) — `population_peri_event_histogram` (`n_units` at `:455`, `resolve_unit_ids` at `:461`, `enumerate(spike_trains)` at `:481`). Task 8.
- [src/neurospatial/behavior/vte.py:617-760](../../../../src/neurospatial/behavior/vte.py) — `compute_vte_session` (trial mask at `:688-690`, window from the whole session at `:714-716`, window record at `:737`). Task 9.
- [src/neurospatial/behavior/navigation.py:1732-1842](../../../../src/neurospatial/behavior/navigation.py) — `instantaneous_goal_alignment` (Returns `:1754-1756`, heading call `:1780`) and `goal_bias` (`min_speed` text `:1809`). Task 10.
- [src/neurospatial/behavior/decisions.py:402-485](../../../../src/neurospatial/behavior/decisions.py) — `pre_decision_heading_stats` (`min_speed` text `:417-418`, NaN filter `:468-469`). Task 11.
- [src/neurospatial/io/nwb/_environment.py:1040-1060, 1126-1179](../../../../src/neurospatial/io/nwb/_environment.py) — `environment_from_position`'s `units` parameter and `_get_position_units` (silent-fallback Notes `:1149-1158`, return `:1178-1179`). Task 12.
- **Files Phase 1 already changed:** `io/nwb/_environment.py` (Phase 1 Task 10 rewrote the writer, reader and `_ReconstructedLayout`; Tasks 3 and 12 here edit only `_get_position_units` and `environment_from_position`), `stats/circular.py` (Phase 1 Task 4 changed three p-values; Task 2 here edits only the plot setup), `environment/factories.py` (Phase 1 Task 5 docstring only), `ops/diffusion.py` (Phase 1 Task 5 docstring only; Tasks 5 and 7 here only call its helpers), and `CHANGELOG.md` (Phase 1 added a `### Fixed` section under `[Unreleased]`; append to it). Re-locate line numbers by symbol.

**Contracts referenced:**

- [Error-message contract](shared-contracts.md#error-message-contract) — every new raise or warning in Tasks 3, 4 and 12 states what was wrong, why it matters, and a `Fix:` line.
- [Input conventions → Population identity](shared-contracts.md#input-conventions) — Task 8: a labelled `TsGroup`'s index becomes `unit_ids`. Labels are never overridden: a `unit_ids=` passed with a labelled group must equal its index, or the call raises (Phase 1 Task 7's `resolve_unit_ids(..., input_ids=)`).

**Designs referenced:** none (the calculus operator is specified inline in Task 5 and the basis generator in Task 7).

## Tasks

Each task is one commit with its regression tests and docstrings. **Proving a test fails before the fix:** write the regression test first and run it against the unmodified code (`uv run pytest <nodeid> -n 0`). Confirm it fails and quote the failure in the commit body. Then implement the fix and confirm the test passes. Do not use `git stash` for this.

Tasks 6–12 were each verified on this branch (`da631a47`) with a minimal probe before being added; the measured numbers are quoted in each task and reused as the regression-test expectations.

1. **Linearize with `edge_id` equal to the enumeration index (`layout/engines/graph.py`).** The root cause is the ordering of `edge_id`, not the type of the node labels: integer labels fail identically. Fix it at the one place every graph environment passes through (`from_graph`, `maze`, `linear_track`, and `from_file` rebuilds). At the top of `build`, after the argument checks (`:106-111`), add:

   ```python
   # track_linearization finds a point's nearest segment by its position in
   # ``graph.edges()`` but looks that segment up by its ``edge_id`` attribute,
   # so the two must coincide. Linearize a copy numbered by enumeration; the
   # caller's graph is never mutated.
   track_graph = graph_definition.copy()
   for enumeration_index, (u, v) in enumerate(track_graph.edges()):
       track_graph.edges[u, v]["edge_id"] = enumeration_index
   self._build_params_used["graph_definition"] = track_graph
   ```

   Then use `track_graph` in place of `graph_definition` at `:115`, `:125` and `:131`. That way `_get_graph_bins` (which numbers segments by enumeration, `helpers/graph.py:116-118`), `to_linear`, `plot` and `ops/diffusion.py:1403` (`_graph_fv`) all read the same graph.

   **Old path removed in the same commit:** factories no longer assign `edge_id`.
   - `_add_edge_with_distance` (`factories.py:64-75`) loses its `edge_id` parameter.
   - Drop the `edge_id` counters at `:127-186` and the `edge_id=i` at `:740`.
   - In `maze(track_graph=...)` (`:915-932`), keep only the `distance` fill.
   - Replace the two false comments (`:66-71` docstring and `:919-924`) with one sentence: "`GraphLayout.build` numbers `edge_id` by `graph.edges()` order; any `edge_id` on the input graph is ignored."
   - Add that sentence to the `graph` parameter of `from_graph` (`:986-988`).

2. **Polar plots follow the library's angle conventions.**
   - **Egocentric** (`egocentric.py:2322-2323`). The convention is 0 = ahead and +π/2 = left (CLAUDE.md; `environment/polar.py:152-153`). Keep `set_theta_zero_location("N")`, set `set_theta_direction(1)`, and fix the comment ("counter-clockwise: +π/2 = left of the animal is drawn on the left"). On `main` a field at +π/2 is drawn at display dx = +80.3 px, which is right of centre.
   - **Allocentric head direction** (`directional.py:2387-2388`). The data convention is 0 = East, π/2 = North (`directional.py:1667-1668`; `ops/egocentric.py` `heading_from_velocity`). The current North-up, clockwise axes therefore draw a North-preferring cell pointing East: π/2 lands at dx > 0, dy ≈ 0. Set `set_theta_zero_location("E")` and `set_theta_direction(1)` explicitly, so a caller-supplied axis is also normalized. Rewrite the docstring sentences at `:2309-2311` and the Notes at `:2340-2347`: "angles are drawn as in the arena, 0 = East (right), π/2 = North (up), counter-clockwise".
   - **`plot_circular_basis_tuning`** (`circular.py:1797-1800`). Same change and same evidence: its own example fits head-direction data (`:1688`), and `circular_basis_metrics` returns `arctan2(beta_sin, beta_cos)` in the same math convention. Add a Notes line stating the orientation.

3. **NWB position readers apply `conversion` and `offset` and normalize unit names.** NWB defines the value in `unit` as `data * conversion + offset`.
   - In `io/nwb/_adapters.py`, next to `timestamps_from_series`, add:

     ```python
     def scaling_from_series(series: Any) -> tuple[float, float]:
         """(conversion, offset) mapping stored values to ``series.unit``."""
         return float(getattr(series, "conversion", 1.0)), float(getattr(series, "offset", 0.0))


     def data_from_series(series: Any) -> NDArray[np.float64]:
         """Materialize ``series.data`` in its declared unit (``data * conversion + offset``)."""
         data = np.asarray(series.data[:], dtype=np.float64)
         conversion, offset = scaling_from_series(series)
         return data * conversion + offset if (conversion, offset) != (1.0, 0.0) else data


     def require_unscaled_for_lazy(series: Any, *, context: str) -> None:
         """Refuse a lazy handle whose stored values are not in ``series.unit``."""
         conversion, offset = scaling_from_series(series)
         if (conversion, offset) != (1.0, 0.0):
             raise ValueError(
                 f"{context}(lazy=True) would return the stored values of '{series.name}', "
                 f"but the series declares conversion={conversion} and offset={offset}, so "
                 f"stored values are not in '{series.unit}'.\n"
                 f"Fix: call {context}(..., lazy=False); values are converted on read."
             )
     ```

   - Use `data_from_series` at `_behavior.py:141` (`read_position`), `_behavior.py:364` (`read_head_direction`, before the degree→radian step) and `_pose.py:123` (`read_pose`, per body part). Call `require_unscaled_for_lazy` on the lazy branches (`_behavior.py:122-138`; `_pose.py:116-120`, per series). `Session.from_nwb` (`recording.py:456`) and the NWB overlays (`_overlays.py:73, 150`) read eagerly, so they inherit the fix.
   - **Unit names.** In `_environment.py`, add a module mapping `_NWB_UNIT_ALIASES = {"meter": "m", "meters": "m", "metre": "m", "metres": "m", "m": "m", "centimeter": "cm", "centimeters": "cm", "cm": "cm", "millimeter": "mm", "millimeters": "mm", "mm": "mm", "pixel": "px", "pixels": "px", "px": "px"}`. `_get_position_units` (`:1179`) returns `_NWB_UNIT_ALIASES.get(unit.strip().lower(), unit)`. NWB's default `"meters"` then becomes the registry value `"m"` (`environment/core.py:398-400`) instead of a free-form string that triggers a warning. Task 12 adds the empty-unit warning to the same function.
   - **Docstrings.** `read_position`, `read_head_direction` and `read_pose` Returns sections, plus the `lazy` parameter: "values are in the series' `unit`: stored × `conversion` + `offset`; `lazy=True` raises when that map is not the identity". `environment_from_position` `units`: "auto-detected from the series `unit`, with NWB long names mapped to `m` / `cm` / `mm` / `px`".

4. **Simulated place-field width uses a linear bin spacing.**
   - At `place_cells.py:201-207`, replace `3.0 * np.mean(env.bin_sizes)` with three times the median nearest-neighbour distance between bin centres. That equals `bin_size` on a regular grid and the bin length on a linearized track. Reuse `neurospatial.ops.binning._estimate_typical_bin_spacing(cKDTree(env.bin_centers), env.bin_centers)`.
   - When the spacing is not finite (a single-bin env), raise `ValueError` with `Fix: pass width=<sigma in environment units>`.
   - Apply the same spacing at `validation.py:217-220`: `max_center_error = 2.0 * spacing`.
   - Correct the docstrings: `place_cells.py:30`, `validation.py:59` and `:163`, `simulation/session.py:192`, `simulation/examples.py:43`. Each becomes "3 × bin spacing (median distance between neighbouring bin centres; equals `bin_size` on a regular grid)".
   - **Old test removed:** `tests/simulation/test_models.py:68-74` (`test_default_width`) asserts the buggy `3 * mean(bin_sizes)`. Replace it.

5. **Physically scaled discrete calculus (`ops/calculus.py`).** On `main`, `D[·, e] = ∓√d_e`, so `abs(gradient(x)) = d**1.5` (1.0, 2.83, 8.0 at bin sizes 1, 2, 4). `div(grad(·))` is then the distance-weighted combinatorial Laplacian, the sign of `divergence` contradicts its docstring, and the goal bin of `−grad(distance)` comes out as a *source* (positive divergence at every bin size).

   Replace this with the mimetic finite-volume pair built on the geometry `env.smooth` uses. For edge `e = (i → j)`, take the length `d_e` and face measure `A_e` from `_finite_volume_geometry(env)`, and the volumes `M` from the same call:

   - `gradient(f)_e = (f_j − f_i) / d_e`, in field units per length unit;
   - `divergence(q)_i = −(1/M_i) Σ_e B_ie A_e q_e`, where `B` is the oriented incidence matrix. This is positive at sources.
   - Then `divergence(gradient(f)) = −M⁻¹(Deg − W) f = −L f` exactly, with diffusion's `W = A/d`. The continuum limit is ∇²f. `⟨grad f, q⟩_{A·d} = −⟨f, div q⟩_M` (discrete Gauss–Green).

   ```python
   def _fv_edges(env: Environment) -> tuple[nx.Graph, NDArray[np.float64]]:
       from neurospatial.ops.diffusion import _finite_volume_geometry

       return _finite_volume_geometry(env)  # graph copy (same edge order) + volumes


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

   The distances must come from the finite-volume copy, not from `env.connectivity`. Otherwise graph environments lose consistency with the diffusion generator at track junctions. `gradient` keeps `diff_op.T @ field`. `divergence` uses `env._divergence_operator_cached @ edge_field`. Add `_divergence_operator_cached` next to `_differential_operator_cached` (`core.py:1112-1118`), also as a `versioned_cached_property`. Edge order is `env.connectivity.edges()` order, which `_finite_volume_geometry`'s `.copy()` preserves (asserted by a test). Layouts without finite-volume geometry raise the same `NotImplementedError` that `env.smooth` raises.

   **Callers and what changes for them.** No `src/` module calls `gradient` or `divergence`. `compute_differential_operator` is called only by `core.py:1118`, and it remains public in `ops.__all__` with the same signature.
   - **Values change for every user:** gradients are now in field units per length unit, and `div(grad)` changes sign and units, from Hz·cm to Hz/cm².
   - **Update the docstrings that state the old identity `L = D @ D.T`:** the module docstring (`calculus.py:1-28`), `compute_differential_operator` (`:51-108`, including its doctest comparing with `nx.laplacian_matrix`), `gradient` (`:173-176`, `:213-226`), `divergence` (`:273-276`, `:339-351`), and `get_differential_operator` (`core.py:1069-1078`).
   - **Rewrite the tests that encode the old maths** in `tests/ops/test_differential.py` at `:32-47`, `:60-74`, `:86-107`, `:156-185`, `:239-260` and `:311-333`. `docs/user-guide/rl-primitives.md:467` is a comment only and needs no change.
   - `ops/basis.py:722-724, 902-904` builds `nx.laplacian_matrix(weight="distance")` independently and is **not** a caller. Task 7 fixes it separately.

6. **Heading interpolation follows the shorter arc, uniformly in angle (`ops/egocentric.py:820-862`).**
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

7. **Graph bases use the finite-volume diffusion generator (`ops/basis.py`).**
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

     The same probe with this generator (`p2extra/b_basis_fix.py`) gives σ = 1.4142 bins at scale 1 and 2.8284 bins at scale 4, for bin sizes 1, 2 and 5, along both the axis and the diagonal. In `heat_kernel_wavelet_basis`, replace `:722-724` with `laplacian = _diffusion_generator(env)`. Layouts without finite-volume geometry raise the same `NotImplementedError` that `env.smooth` raises.
   - **Docstrings (heat kernel).** `scales`: "diffusion time in units of bin spacing²; the kernel's standard deviation is `sqrt(2·scale)` bins (0.5 → 1, 1 → 1.41, 2 → 2, 4 → 2.83 bins), independent of `bin_size`". Replace the "~N bins radius" table, the "Adjust if your bin_size is unusual" advice and the "Roughly" relation with that statement. Rewrite "Laplacian weighting" (`:654-657`) as "uses the finite-volume generator of `env.smooth` (face measure over centre distance), so spread is isotropic and in bin units". Update the module-docstring line (`:36`) and the out-of-range error text at `:707-714` to the same numbers.
   - **Chebyshev: deliberately unchanged, documented.** `chebyshev_filter_basis` rescales by `λ_max` (`L_scaled = 2L/λ_max − I`), so a global change of weight scale has no effect. Its documented contract is hop locality: "nonzero only for bins within k steps", "max_degree ≈ desired_radius / bin_size". That holds on this branch (degree-3 support radius = 3 bins, probe `p2extra/b_basis_aniso.py`). Switching it to the finite-volume generator would drop corner-only (diagonal) steps on Cartesian grids, where `A = 0` (`diffusion.py:_cartesian_fv`), and change that contract. Instead, add a Notes sentence: "The operator is the connectivity graph's Laplacian with `distance` edge weights, rescaled by its spectral radius. It defines hop locality only and is not a physical diffusion; use `heat_kernel_wavelet_basis` for bin-size-independent spread."
   - **Tests to update.** `tests/ops/test_basis.py` heat-kernel tests that depend on the old spread (search `heat_kernel_wavelet_basis`). The `_estimate_spectral_radius` tests at `:454-500` build their own distance Laplacian and are unaffected.

8. **`population_peri_event_histogram` accepts a pynapple `TsGroup` (`events/alignment.py:343`).**
   - **Bug (verified).** The loop `for unit_idx, spike_times in enumerate(spike_trains)` (`:481`) iterates a `TsGroup`'s *keys*. Probe `p2extra/c_peth.py`, three units with keys `3, 7, 9`: a real `nap.TsGroup` and the `UserDict` double both raise `AxisError: axis -1 is out of bounds for array of dimension 0`. The same trains as a list give mean rates `[4.06, 10.0, 2.08]` Hz.
   - **Fix.** Mirror `compute_spatial_rates` (`encoding/spatial.py:3408-3419`) and Phase 1 Task 7: `trains, extracted_ids = as_spike_trains_with_ids(spike_trains)` before the empty check; use `trains` everywhere below; call `resolve_unit_ids(unit_ids, n_units, input_ids=extracted_ids, context="population_peri_event_histogram")` (`:461`), which Phase 1 Task 7 added and which raises when both are given and differ. Document in the `spike_trains` parameter that a `TsGroup` is accepted, that its index becomes `unit_ids`, and that a `unit_ids=` passed with it must equal that index. (Phase 6 renames this parameter to `spike_times`; Phase 3b adds `epochs`/`spike_window` to the same function.)

9. **`compute_vte_session` clamps each pre-decision window to its trial (`behavior/vte.py:714-737`).**
   - **Bug (verified).** The entry time is found within the trial (`:688-690`), but the window is cut from the **whole session** (`extract_pre_decision_window(positions, times, …)` at `:714`), so it reaches into the inter-trial interval or the previous trial. Probe `p2extra/d_vte.py`: a trial starting at 5.0 s that enters the decision region at 5.267 s gets the window `[4.267, 5.267]`. Its head sweep is **7.689 rad**, all of it from pre-trial zig-zagging. The same window clamped to `[5.0, 5.267]` gives 0.000 rad.
   - **Fix.** Extract the window from the trial's samples: `extract_pre_decision_window(trial_positions, trial_times, entry_time, window_duration)`. Record `(max(entry_time - window_duration, trial.start_time), entry_time)` at `:737`. Windows that become shorter than 3 samples are skipped by the existing rule (`:717-719`). Docstring: `window_duration` "is clipped at the trial start; samples before `trial.start_time` are never used". `compute_vte_trial` has no trial bounds and is unchanged.

10. **Goal alignment excludes stationary samples, as documented (`behavior/navigation.py:1732-1842`).**
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

11. **`pre_decision_heading_stats` excludes stationary samples, as documented (`behavior/decisions.py:402-485`).**
    - **Bug (verified).** The NaN filter at `:468-469` ("Filter out NaN (stationary periods)") never fires unless *every* sample is stationary, because `heading_from_velocity` has already interpolated the stationary headings. Probe `p2extra/f_predec.py`, the same east/stop/north path: on this branch `(mean_direction, circular_variance, mrl) = (0.7854, 0.1849, 0.8151)`; over the 20 moving samples only it is `(0.7854, 0.2929, 0.7071)`.
    - **Fix.** Use Task 10's helper: `heading, speed = _velocity_heading_and_speed(positions, dt)`, then `valid_headings = heading[speed >= min_speed]`. The existing all-stationary return `(0.0, 1.0, 0.0)` is unchanged.

12. **`environment_from_position` warns when it assumes `"cm"` (`io/nwb/_environment.py:1126-1179`).** This edits the same `_get_position_units` as Task 3 and lands after it.
    - **Bug (verified).** Probe `p2extra/g_nwbunit.py`: a `SpatialSeries` with `unit=""` gives `env.units == "cm"` and **no warning**. The Notes call this "a silent fallback". (pynwb fills `"meters"` when `unit` is omitted, so this case is an explicitly empty unit.) Non-empty unrecognized units (`"a.u."`, `"inches"`) already warn through the `env.units` registry check (`environment/core.py:398-425`), which names the value, so they need no change.
    - **Fix.** When `spatial_series.unit` is empty or None, emit one `UserWarning` and still return `"cm"`:

      > Position series '{name}' declares no unit (unit={unit!r}); assuming 'cm'. A wrong unit mislabels every distance, speed and bin size derived from this environment.
      > Fix: pass units='cm' (or 'm', 'mm', 'px') to environment_from_position to state the real unit.

      `environment_from_position` calls `_get_position_units` only when `units is None` (`:1108`), so passing `units=` silences it. Replace the "silent fallback" Notes (`:1149-1158`) with "warns, then assumes `cm`", and add the same to the `units` parameter (`:1050-1052`).

13. **User-facing documentation.**
    - Under `## [Unreleased]` → `### Fixed` in `CHANGELOG.md`, add one bullet per Tasks 1–12, each with the symptom and the numbers from the Validation slice. Mark these as **behavior changes**:
      - Task 5: the new units, the sign of `divergence`, and `div(grad) = ∇²` (negative semidefinite);
      - Task 7: heat-kernel `scales` now give `sqrt(2·scale)` bins at every bin size;
      - Task 8: `population_peri_event_histogram` accepts a `TsGroup` and takes its `unit_ids`; a conflicting `unit_ids=` raises;
      - Task 9: VTE windows stop at the trial start;
      - Tasks 10–11: goal alignment and pre-decision heading statistics exclude samples below `min_speed`;
      - Task 12: the new warning.
    - Update `examples/09_differential_operators.py`. Change the operator formulas in the markdown at `:46-49`, `:127`, `:246` and `:326`. Replace the "equals NetworkX's Laplacian" verification (`:425-453`) with a check that `divergence(gradient(x² + y²))` is 4 in interior bins. The goal-sink claim at `:271` becomes true.
    - Then run `uv run jupytext --sync examples/09_differential_operators.py`, re-execute the notebook so its stored outputs match (`mkdocs.yml:93` has `execute: false`), and run `uv run python docs/sync_notebooks.py`. CI checks `git diff --exit-code docs/examples`.

## Deliberately not in this phase

- **Object-vector-cell classification bias and an allocentric object-vector mode** are Phase 5, and so are the Høydal 2019 citation fixes.
- **The ported archive fixes,** including NWB environment *geometry* persistence, are Phase 1. Task 3 here edits other functions of the same `_environment.py`, so rebase on Phase 1 if it merged first.
- **Gap handling, errors and executable docs, API surface, output polish** are Phases 3, 4, 6 and 7.
- **Changing `chebyshev_filter_basis`'s operator.** Task 7 documents why it keeps the distance-weighted graph Laplacian (hop locality is its contract, and a global weight scale cancels in its rescaling).
- **Gap-aware (per-run) kinematics.** Tasks 6, 10 and 11 fix single-recording behaviour. Phase 3c later runs `heading_from_velocity`, its interpolation and Task 10's speed mask per valid run, with `times` in place of `dt`.
- **A second warning for non-empty unrecognized NWB units.** The `env.units` registry check already warns and names the value (Task 12 probe).
- **The labelled-group handling in `peri_event_histogram` (single unit).** It takes one 1-D train, so there are no keys to mis-iterate.
- **Upstream `track_linearization`.** Its positional-index-as-`edge_id` lookup should be reported upstream; Task 1 does not depend on that.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/environment/test_factory_presets.py::test_w_maze_to_linear_matches_track_distance[str labels, int labels]` | W maze `bl(0,0) bm(50,0) br(100,0) al(0,50) am(50,50) ar(100,50)`, bin 5: `to_linear` of `(25,0),(75,0),(0,40),(50,25),(100,25),(100,45)` == `[25, 75, 140, 175, 225, 245]` (`main`: `[25, 175, 114.03, 175, 225, 245]`) |
| `…::test_w_maze_to_linear_agrees_with_bin_at` | 2000 on-track samples (seed 0, t∈[0.02, 0.98] per edge): fraction where `linear_point_to_bin_ind(to_linear(x)) != bin_at(x)` == 0.0 (`main`: 0.3775) |
| `…::test_from_graph_ignores_input_edge_ids` | the same graph via `from_graph` with `edge_id`s following `edge_order` gives the correct values above; the caller's `edge_id`s are unchanged after the call |
| `…::test_w_maze_to_linear_survives_file_roundtrip` | `to_file` then `from_file`, then the same six expected values (`main`: 175 and 114.03) |
| `…::test_plus_and_t_maze_to_linear_unchanged` (guard) | plus and T mazes give identical `to_linear` before and after |
| `tests/encoding/test_encoding_egocentric.py::test_object_vector_plot_draws_left_on_left` | `EgocentricRateResult` peaked at +π/2 (polar env 0–50, 10×12 bins): the peak marker's display dx < 0 (`main`: +80.3 px) |
| `tests/encoding/test_encoding_directional.py::test_head_direction_plot_north_is_up` | after `plot_head_direction_tuning`, `ax.transData` maps (π/2, r) to dx≈0, dy>0 and (0, r) to dx>0, dy≈0 (`main`: π/2 at dx>0, dy≈0) |
| `tests/stats/test_stats_circular.py::test_circular_basis_plot_north_is_up` | `plot_circular_basis_tuning(1.0, 0.0)`: (π/2, r) maps to dx≈0, dy>0 (`main`: dx = +248 px, dy = 0) |
| `tests/nwb/test_behavior.py::test_read_position_applies_conversion_and_offset` | pixel data 0–500, `unit="meters"`, `conversion=0.002`, `offset=0.1` → `positions.min(0) == [0.1, 0.1]`, `positions.max(0) == [1.1, 1.1]` (`main`: max `[500, 500]`) |
| `…::test_read_position_lazy_refuses_scaled_series` | the same series with `lazy=True` → `ValueError` containing `Fix:`; an identity-scaled series stays lazy (guard) |
| `…::test_read_head_direction_applies_conversion` | degrees stored ×0.5 with `conversion=2.0` → radians equal `deg2rad(2·stored)` |
| `tests/nwb/test_pose.py::test_read_pose_applies_conversion_and_offset` | body-part coordinates equal stored × conversion + offset |
| `tests/nwb/test_environment.py::test_environment_from_position_uses_converted_meters` | `env.units == "m"`, with no units-registry warning; `bin_centers` lie within [0.1 − 0.05, 1.1 + 0.05] (`main`: units `"meters"`, extent 500 × 500) |
| `tests/simulation/test_models.py::test_default_width_is_three_bin_spacings[1,2,5]` | `from_samples` grid on 0–100 cm: `width == 3 * np.diff(env.layout.grid_edges[0])[0]` at rtol 1e-12, and ≈ 3·bin_size within 5% (`main`: 3, 12, 75 cm for bin 1, 2, 5) |
| `…::test_default_width_on_track_uses_bin_length` | W-maze env with bin 5 → `width` ≈ 15 |
| `…::test_default_width_single_bin_raises` | 1-bin env, `width=None` → `ValueError` with `Fix: pass width=` |
| `tests/simulation/test_validation_sim.py::test_default_center_error_threshold` | the default `max_center_error` equals 2 × spacing (4.0 cm at bin 2; `main`: 8.0) |
| `tests/ops/test_differential.py::test_gradient_of_x_is_unit_slope[1,2,4]` | `from_grid_mask` 20×20 cm with edges `np.arange(0, 20+h/2, h)`: on x-axis edges, `np.abs(gradient(env, x)) == 1.0` at atol 1e-12 (`main`: 1.0, 2.8284, 8.0) |
| `…::test_div_grad_quadratic_is_continuum_laplacian[1,2,4]` | `divergence(gradient(x²+y²)) == 4.0` on interior (degree-8) bins (`main`: −15.31, −122.51, −980.08) |
| `…::test_div_grad_equals_negative_diffusion_generator[grid, hex, W-maze graph]` | equals `−(M⁻¹(Deg − W)) f` built from `diffusion._assemble_W(_finite_volume_geometry(env)[0])` for random f, at rtol 1e-10 |
| `…::test_divergence_is_negative_adjoint_of_gradient` | `Σ A·d·grad(f)·q == −Σ M·f·div(q)` at rtol 1e-10 |
| `…::test_goal_is_a_sink` | goal = bin nearest the arena centre, h ∈ {1, 2, 4}: `divergence(−gradient(dist_to_goal))[goal] < 0` (`main`: positive at every h, +12 for the (10.5, 10.5) bin at h = 1) |
| `…::test_operator_edge_order_matches_connectivity` | the finite-volume copy's `list(edges())` equals `list(env.connectivity.edges())`; `gradient(f)[k]` equals `(f[v] − f[u]) / d` for the k-th connectivity edge |
| `uv run pytest --doctest-modules src/neurospatial/ops/calculus.py src/neurospatial/environment/core.py` | the rewritten doctests pass |
| `tests/ops/test_reference_frames.py::test_heading_interpolation_is_uniform_on_shorter_arc[π/2, π−0.1, π]` | 3 masked samples between valid `0` and `turn`: the filled values equal `turn · [0.25, 0.5, 0.75]` at atol 1e-12, with no value equal to an endpoint (before the fix at π − 0.1: `[0.050, 1.521, 2.992]`) |
| `…::test_heading_interpolation_across_pi_wrap` | valid `π − 0.1` and `−π + 0.1` with one masked sample between → `π` (or `−π`) exactly within atol 1e-12, never near 0 (shorter arc through ±π) |
| `…::test_antipodal_turn_has_no_jump_to_zero` | valid `π/2` and `−π/2` with 3 masked samples → `[π/4, 0, −π/4]`, monotone (before the fix: `[π/2, 0, −π/2]`, a step) |
| `…::test_heading_from_velocity_turn_is_gradual` | east → 3 slow samples → west, `min_speed=0.5`: consecutive headings differ by ≤ π/4 + 1e-12 (before the fix: a jump of π/2 then π/2 at the midpoint) |
| `…::test_nan_anchor_not_propagated` | an unmasked NaN heading next to a masked run leaves the masked values finite |
| `tests/ops/test_basis.py::test_heat_kernel_spread_is_bin_size_independent[1,2,5]` | 40 × 40-bin `from_grid_mask` grid, centre bin, `normalize="none"`: σ along x and along the diagonal equals `sqrt(2·scale)` bins at rtol 1e-3 for `scale ∈ {1, 4}` (before the fix: 2.77, 3.91, 6.17 bins at scale 1) |
| `…::test_heat_kernel_uses_smooth_generator` | the basis column for a centre equals `expm_multiply(-s · spacing² · M⁻¹(Deg − W), δ)` built from `diffusion._assemble_W(_finite_volume_geometry(env)[0])` at rtol 1e-10, on a hexagonal env and the W maze |
| `…::test_chebyshev_hop_locality_unchanged` (guard) | degree-k column is zero beyond k connectivity hops and identical to the pre-change output (rtol 1e-12) |
| `tests/events/test_alignment.py::test_population_psth_accepts_tsgroup` | spike-group double with keys `3, 7, 9` → `unit_ids == [3, 7, 9]` and `firing_rates` equal to the list input's (mean rates `[4.06, 10.0, 2.08]` Hz on the probe data); a real `nap.TsGroup` gives the same (`@pytest.mark.pynapple`). `unit_ids=[3, 7, 9]` is accepted; `unit_ids=[9, 7, 3]` raises `ValueError` listing both orders. Before the fix: `AxisError` |
| `tests/behavior/test_vte.py::test_session_window_clamped_to_trial_start` | the Task 9 probe path: `trial_results[0].window_start == 5.0` and `head_sweep_magnitude == 0.0` within 1e-12 (before the fix: window start 4.267, head sweep 7.689 rad) |
| `tests/behavior/test_behavior_navigation.py::test_goal_alignment_nan_when_stationary` | the Task 10 path: all 30 stationary samples are NaN and moving samples are finite; `goal_bias == 0.5055` at atol 1e-4 (before the fix: 0 NaN, 0.5862) |
| `…::test_goal_bias_all_stationary_is_nan` (guard) | a motionless trajectory → `goal_bias` is NaN without raising |
| `tests/behavior/test_decision_analysis.py::test_pre_decision_heading_stats_excludes_stationary` | the same path: `(0.7854, 0.2929, 0.7071)` at atol 1e-4 (before the fix: `(0.7854, 0.1849, 0.8151)`) |
| `tests/nwb/test_environment.py::test_empty_unit_warns_and_assumes_cm` | `unit=""` → one `UserWarning` containing `"declares no unit"` and `"Fix: pass units="`; `env.units == "cm"`. Passing `units="cm"` emits no warning (before the fix: silent) |

Run the slice with `-n 0`. Then run `uv run pytest -m "not slow and not napari"`, `uv run pytest tests/nwb -n 0` (under `--extra nwb`), and `uv run mypy src/neurospatial/`.

## Fixtures

- **W maze, plus maze and T maze.** Built inline from the node coordinates in the table. Reuse `tests/conftest.py`'s session-scoped `tmaze_env` where it fits.
- **NWB.** In-memory `NWBFile` with a `Position` `SpatialSeries`, as in the triage repro `claim6_conversion_offset.py`: 100 rows cycling `(0,0),(500,0),(500,500),(0,500),(250,250)` at 30 Hz. Add a `CompassDirection` series and an ndx-pose series with non-identity `conversion`. These run in `test_nwb.yml`.
- **Polar plots.** Use the `Agg` backend. Get display offsets from `ax.transData.transform` relative to `(0, 0)`. Build the egocentric field with `Environment.from_polar_egocentric((0, 50), (−π, π), 5.0, 2π/12)` and a von-Mises-in-angle × Gaussian-in-distance rate peaked at (25, +π/2).
- **Calculus.** `Environment.from_grid_mask(np.ones((n, n), bool), (e, e))` with exactly uniform edges, so that `d` equals `h` exactly. Use a hexagonal `from_samples` env (seed 0) and the W maze for the generator-equivalence test.
- **Basis.** The same `from_grid_mask` construction with `edges = np.arange(41) * h`, centre bin = the bin nearest `(20h, 20h)`; the hexagonal env and W maze above.
- **Headings, VTE, goal alignment.** Inline paths copied from the probes: 10 Hz `t = np.arange(50) * 0.1`, `x = np.minimum(t, 1) * 20`, `y = np.maximum(t - 4, 0) * 20` (east, stop, north; goal `(1000, 20)`); the VTE path is `t = np.arange(0, 10, 1/30)`, zig-zag before 5 s, then east at 30 cm/s into `box(45, 40, 60, 60)`, one `Trial(5.0, 9.9, …)`.
- **Population PETH.** Three seeded uniform trains (seed 0; 500, 1000 and 200 spikes over 100 s) with keys `3, 7, 9`, events every 5 s from 5 to 90 s, `window=(-1, 1)`, `bin_size=0.1`. Reuse Phase 1's `make_spike_group` fixture.
- **NWB empty unit.** The in-memory `NWBFile` above with `SpatialSeries(unit="")`.
- **Probe scripts** live in the session scratchpad (`p2extra/`); the tests reproduce them inline and do not depend on those files.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
