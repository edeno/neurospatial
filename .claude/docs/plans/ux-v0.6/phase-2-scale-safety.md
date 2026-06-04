# Phase 2 — Scale Safety (1000s of neurons, long sessions)

**Goal:** Remove the memory cliffs and the compute redundancy that bite at scale, and stop the docs teaching the per-neuron loop. Additive + perf; no API removals. Return-type stability preserved (constraint B).

**Acceptance for the phase:** memory/throughput benchmarks committed (small fixtures + asserted bounds); numeric parity tests vs current behavior; `ruff`/`mypy` clean.

---

### 2.1 — Memory-safe decoding via a *separate focused function* (no contract break)  ⚑ headline
- **Files:** `decoding/posterior.py` (`decode_position`, `normalize_to_posterior` `:153,218,221`), `decoding/_result.py` (`DecodingResult` `:91` + **new `DecodingSummary`**).
- **Problem:** dense float64 `(n_time, n_bins)` posterior always materialized (`posterior.py:518`) with a **3–4× transient peak** from `.copy()`+`exp`.
- **Finding #1 (do NOT make the posterior lazy):** `DecodingResult.posterior` is assumed to be a real ndarray throughout — `_result.py:91,104,133,177,334` use `.shape`, `argmax`, matmul, entropy, plotting, xarray export. A `posterior="lazy"` flag setting `.posterior=None` (or a recompute-proxy) would silently break all of those. So the memory-safe path is a **separate function with its own return type**, not a flag on `decode_position`.
- **Design — two focused functions, each return-type-stable** (the SciPy "refactor mode-flags into multiple focused functions" remedy):
  - **`decode_position(...) -> DecodingResult`** — contract **unchanged**: `.posterior` is always a fully-materialized ndarray and every existing method keeps working. Add **`dtype: float32|float64 = float64`** (halves stored + transient memory) and **`time_chunk: int | None`** (compute `exp`/normalize in time-blocks into a preallocated `(n_time, n_bins)` output, cutting the transient peak from ~3–4× to ~1×). No flag changes the return type.
  - **`decode_position_summary(...) -> DecodingSummary`** (NEW) — streams over time chunks and **never materializes the full `(n_time, n_bins)` posterior**; keeps only `(n_time, …)` reductions. `DecodingSummary` is a frozen dataclass: `times`, `map_position`, `mean_position`, `posterior_entropy`, `peak_prob`, `env`, plus the standard terminal verbs (`to_dataframe`/`summary`/`plot`/`to_xarray` → a track `Dataset` with a `time` dim and no `bin` posterior axis). It **shares accessor names** with `DecodingResult` so user code ports between them. Add a thin `decode_session_summary(...)` golden-path wrapper (sibling of `decode_session`).
  - Both result types are shaped so a future chunked/dask/zarr posterior can slot in without an API change (design-for-chunking-now).
- **Targets:** 1 hr / 25 ms / 5000 bins: `decode_position_summary` ≤ ~few hundred MB (never allocates `(144k, 5000)`); `decode_position(dtype=float32, time_chunk=N)` ≈ half the float64 stored size with transient peak ≈ 1×.
- **Tests:** `decode_position_summary` never allocates `(n_time, n_bins)` (trace/peak-memory assert on a fixture) and its MAP equals `decode_position(...).map_position`; `DecodingResult.posterior` stays a real ndarray and all its methods pass unchanged; `dtype=float32` within tol; `time_chunk` parity; `decode_session_summary` forwards correctly.

### 2.2 — Sparse / banded smoothing kernel(s)
- **Two distinct kernels — target the right cache (finding):**
  - **Diffusion kernel** (the **default** `smoothing_method="diffusion_kde"`): built by `env.compute_kernel(...)` and cached on **`env._kernel_cache`** keyed `(bandwidth, mode)` (`environment/fields.py:121,135`). This is the primary scale cliff.
  - **Dense Gaussian-KDE kernel** (`smoothing_method="gaussian_kde"`): the `(n_bins, n_bins)` matrix in `encoding/_smoothing.py:414`, cached in the module-level **weakref** dict (already id-reuse-guarded) and consumed by the batch GEMM `:646-671`.
- **Problem:** both are dense `(n_bins, n_bins)` float64, scaling `n_bins²` independent of neuron count (3.2 GB @ 20k bins); diffusion is intrinsically local (sparse).
- **Change (diffusion = REQUIRED v0.6 target):** sparsify the **diffusion** kernel where it is built (`env.compute_kernel`) and store the sparse (CSR) form on `env._kernel_cache`, **preserving existing env cache semantics** (key `(bandwidth, mode)`, `env.clear_cache()`, `cache=` flag). The batch GEMM `kernel @ spike_counts.T` works transparently with a sparse kernel. Keep a dense fallback for tiny `n_bins`. Require **numeric parity vs dense within tol** on a fixture.
- **Gaussian-KDE sparsification = OPTIONAL / cautious (finding):** truncating the dense `gaussian_kde` kernel (`_smoothing.py`, weakref-guarded) changes numerical results unless the cutoff is carefully defined. Treat it as optional, **gated on BOTH** an explicit parity check (within tol vs dense) **and** a measured memory/perf win. If either fails, leave `gaussian_kde` dense — it is not the default method, so the diffusion win already covers the common path.
- **Targets:** 20k-bin kernel memory 3.2 GB → few MB; smoothing GEMM `O(n_bins²·n_units)` → `O(nnz·n_units)`; no perf regression at typical (≤5k) bins.
- **Tests:** sparse vs dense rate-map parity within tol; memory assertion on a high-bin fixture; cache identity guard still holds.

### 2.3 — Kill double metric recompute + add `n_jobs`
- **Files:** `encoding/spatial.py:1517` (`to_dataframe`) → `:1387` (`detect_cell_types`/`classify` recomputes grid/border scores), `encoding/population.py:287-296` (`population_coverage` per-neuron `detect_place_fields`).
- **Change:** compute grid/border scores once and pass them into `classify(grid_scores=…, border_scores=…)`; thread `n_jobs` through `population_coverage` (joblib, mirroring `bin_spike_trains`).
- **Tests:** grid/border computed once per `to_dataframe()` (call-count assertion); `n_jobs>1` parity with `n_jobs=1`.

### 2.4 — Stop teaching the per-neuron loop
- **Files:** `encoding/population.py:220-225` (docstring example) + `:265` (error message); `compute_spatial_rate` docstring.
- **Change:** rewrite the example/error to `compute_spatial_rates(...)`; add to `compute_spatial_rate`'s docstring: "Computing for many neurons? Use `compute_spatial_rates` — it shares occupancy/kernel work once."
- **Tests:** doc-snippet for the corrected example; lint that the docstring no longer shows a per-neuron `compute_spatial_rate` loop.

### 2.5 — `float32` option for rate maps
- **Files:** `encoding/spatial.py` (`compute_spatial_rates`), `_smoothing.py:671` (final `.astype(np.float64)`).
- **Change:** `dtype: np.float32 | np.float64 = float64` on `compute_spatial_rates`; halves `(n_units, n_bins)` and the downstream decode working set.
- **Tests:** float32 result within tol of float64; dtype honored end-to-end into `decode_session`.

### 2.6 — Expose speed filtering in the batch encode/occupancy path
- **Files:** `encoding/spatial.py` (`compute_spatial_rate`/`compute_spatial_rates`), shared occupancy in `encoding/_binning.py` / `env.occupancy` (which already supports `min_speed`).
- **Problem:** `env.occupancy(..., min_speed=…)` can exclude immobility, but the batch `compute_spatial_rates` computes occupancy internally with **no `min_speed`** — so the batch/decoding path can't drop low-speed samples from rate maps, and `decode_session` therefore can't forward it (Phase 0 §0.7 Note B promises this is tracked here).
- **Change:** add a keyword-only `min_speed: float | None = None` to `compute_spatial_rate(s)`, forwarded into the shared occupancy computation so the batch path matches `env.occupancy`. Then `decode_session` may forward `min_speed`.
- **Tests:** `compute_spatial_rates(..., min_speed=v)` parity with `env.occupancy(min_speed=v)` masking; default `None` leaves current behavior unchanged.

**PR deliverable:** Phase 2 PR with a `benchmarks/` (or `tests/.../test_scaling.py`) asserting the memory/throughput bounds on small fixtures; CHANGELOG `### Added`/`### Performance`.
