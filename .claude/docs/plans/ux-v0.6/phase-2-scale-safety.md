# Phase 2 — Scale Safety (1000s of neurons, long sessions)

**Goal:** Remove the memory cliffs and the compute redundancy that bite at scale, and stop the docs teaching the per-neuron loop. Additive + perf; no API removals. Return-type stability preserved (constraint B).

**Acceptance for the phase:** memory/throughput benchmarks committed (small fixtures + asserted bounds); numeric parity tests vs current behavior; `ruff`/`mypy` clean.

---

### 2.1 — Memory-safe decoding via a *separate focused function* (no contract break)  ⚑ headline
- **Files:** `decoding/posterior.py` (`decode_position`, `normalize_to_posterior` `:153,218,221`), `decoding/_result.py` (`DecodingResult` `:91` + **new `DecodingSummary`**).
- **Problem:** dense float64 `(n_time, n_bins)` posterior always materialized (`posterior.py:518`) with a **3–4× transient peak** from `.copy()`+`exp`.
- **Finding #1 (do NOT make the posterior lazy):** `DecodingResult.posterior` is assumed to be a real ndarray throughout — `_result.py:91,104,133,177,334` use `.shape`, `argmax`, matmul, entropy, plotting, xarray export. A `posterior="lazy"` flag setting `.posterior=None` (or a recompute-proxy) would silently break all of those. So the memory-safe path is a **separate function with its own return type**, not a flag on `decode_position`.
- **Design — two focused functions, each return-type-stable** (the SciPy "refactor mode-flags into multiple focused functions" remedy):
  - **`decode_position(...) -> DecodingResult`** — contract **unchanged**: `.posterior` is always a fully-materialized ndarray and every existing method keeps working. Add **`dtype: float32|float64 = float64`** (halves stored + transient memory) and a **hybrid `time_chunk: int | None`**: `None` (default) keeps the **byte-exact** full-matmul path (transient ~3× over the posterior); an **explicit `time_chunk=N`** computes the Poisson log-likelihood **blockwise into the preallocated `(n_time, n_bins)` output**, cutting the transient peak to **~1×** over the returned posterior. The opt-in path is **tolerance-equal** (~1e-15; MAP/argmax identical; rows sum to 1) to the full path — the per-block likelihood matmul is a different BLAS shape than the full matmul, so it is not byte-exact (consistent with `decode_position_summary`). No flag changes the return type.
  - **`decode_position_summary(...) -> DecodingSummary`** (NEW) — streams over time chunks and **never materializes the full `(n_time, n_bins)` posterior**; keeps only `(n_time, …)` reductions. `DecodingSummary` is a frozen dataclass: `times`, `map_position`, `mean_position`, `posterior_entropy`, `peak_prob`, `env`, plus the standard terminal verbs (`to_dataframe`/`summary`/`plot`/`to_xarray` → a track `Dataset` with a `time` dim and no `bin` posterior axis). It **shares accessor names** with `DecodingResult` so user code ports between them. Add a thin `decode_session_summary(...)` golden-path wrapper (sibling of `decode_session`).
  - Both result types are shaped so a future chunked/dask/zarr posterior can slot in without an API change (design-for-chunking-now).
- **Targets:** 1 hr / 25 ms / 5000 bins: `decode_position_summary` ≤ ~few hundred MB (never allocates `(144k, 5000)`); `decode_position(dtype=float32, time_chunk=N)` — with an **explicit** `time_chunk` (the opt-in blockwise-likelihood path) — ≈ half the float64 stored size with transient peak ≈ 1× over the returned posterior. This opt-in path is **tolerance-equal** (~1e-15, MAP-identical) to the full path, not byte-exact; the **default `time_chunk=None`** keeps the full-matmul **byte-exact** path (transient ~3×).
- **Tests:** `decode_position_summary` never allocates `(n_time, n_bins)` (trace/peak-memory assert on a fixture) and its MAP equals `decode_position(...).map_position`; `DecodingResult.posterior` stays a real ndarray and all its methods pass unchanged; `dtype=float32` within tol; `time_chunk` parity; `decode_session_summary` forwards correctly.

### 2.2 — Smoothing-kernel memory (diffusion is dense by construction — re-scoped, H1+M1)
- **Two distinct kernels, correct locations (M1):**
  - **Diffusion kernel** (default `smoothing_method="diffusion_kde"`): built in `ops/smoothing.py:compute_diffusion_kernels` via `scipy.sparse.linalg.expm(-t·L)` (`ops/smoothing.py:161`), then `.toarray()` (`:165`) + dense column-normalization (`:172-193`). `env.compute_kernel(...)` (`encoding/_smoothing.py:414` → `environment/fields.py:121,135`) only **caches** the result on `env._kernel_cache` keyed `(bandwidth, mode)`.
  - **Dense Gaussian-KDE kernel** (`smoothing_method="gaussian_kde"`): `_get_gaussian_kernel` / module-level weakref `_GAUSSIAN_KERNEL_CACHE` (`encoding/_smoothing.py:78-150`), consumed by the batch GEMM (`:646-671`).
- **Correction (H1) — the diffusion kernel is NOT intrinsically sparse.** The heat kernel `exp(-t·L)` of a *connected* graph is **mathematically dense** (every entry > 0; it only *decays* with geodesic distance) and is **built dense** by `expm` — the code itself says "Matrix exponential is O(n³) and dense" (`ops/smoothing.py:49`). So "store CSR" (a) cannot avoid the O(n²) **build** peak, and (b) truncating below a threshold **changes results**. The earlier "intrinsically sparse / REQUIRED parity-within-tol" framing was wrong and is dropped.
- **Re-scoped change (both kernels: optional, parity+memory-gated):**
  - A truncated-sparse cached form (diffusion or gaussian) is allowed **only if** it (i) holds numeric parity within an explicit, documented tolerance **and** (ii) shows a measured memory/perf win; otherwise leave it dense. The batch GEMM `kernel @ spike_counts.T` works transparently with a sparse kernel.
  - The *real* fix for the dense **build** peak is a different algorithm — `scipy.sparse.linalg.expm_multiply` applied column-block-wise (never materializing the full kernel) or a Chebyshev/Krylov approximation — a **numerical rewrite flagged as a stretch goal**, not a committed v0.6 deliverable.
- **Committed v0.6 deliverable:** prominently **document** the kernel memory cost and gate the high-bin path; the *reliable* memory wins this release are `float32` rate maps (§2.5) and `time_chunk`/`DecodingSummary` decode (§2.1). Do **not** promise a guaranteed 3.2 GB → few-MB diffusion win.
- **Targets:** 20k-bin kernel memory 3.2 GB → few MB; smoothing GEMM `O(n_bins²·n_units)` → `O(nnz·n_units)`; no perf regression at typical (≤5k) bins.
- **Tests:** sparse vs dense rate-map parity within tol; memory assertion on a high-bin fixture; cache identity guard still holds.

### 2.3 — Kill double metric recompute + add `n_jobs`
- **Files:** `encoding/spatial.py:1517` (`to_dataframe`) → `:1387` (`detect_cell_types`/`classify` recomputes grid/border scores), `encoding/population.py:287-296` (`population_coverage` per-neuron `detect_place_fields`).
- **Change:** compute grid/border scores once and pass them into the labeler. **Note (gap):** `detect_cell_types`/`label_cell_types` (`spatial.py:1303`) has **no** `grid_scores=`/`border_scores=` parameters today — add them (additive) so precomputed scores can be injected instead of recomputed. Thread `n_jobs` through `population_coverage` (joblib, mirroring `bin_spike_trains`).
- **Tests:** grid/border computed once per `to_dataframe()` (call-count assertion); `n_jobs>1` parity with `n_jobs=1`.

### 2.4 — Stop teaching the per-neuron loop
- **Files:** `encoding/population.py:220-225` (docstring example) + `:265` (error message); `compute_spatial_rate` docstring.
- **Change:** rewrite the example/error to `compute_spatial_rates(...)`; add to `compute_spatial_rate`'s docstring: "Computing for many neurons? Use `compute_spatial_rates` — it shares occupancy/kernel work once."
- **Tests:** doc-snippet for the corrected example; lint that the docstring no longer shows a per-neuron `compute_spatial_rate` loop.

### 2.5 — `float32` option for rate maps
- **Files:** `encoding/spatial.py` (`compute_spatial_rates`), `_smoothing.py:671` (final `.astype(np.float64)`).
- **Change:** `dtype: np.float32 | np.float64 = float64` on `compute_spatial_rates`; halves `(n_units, n_bins)` and the downstream decode working set.
- **Tests:** float32 result within tol of float64; dtype honored end-to-end into `decode_session`.

### 2.6 — Speed filtering in the batch encode path — must mask BOTH occupancy AND spikes
- **Files:** `encoding/spatial.py` (`compute_spatial_rate`/`compute_spatial_rates`), `encoding/_binning.py` (spike binning `:120,138`; batch occupancy `compute_occupancy` `:323`), `env.occupancy` (`environment/trajectory.py`).
- **Problem (correctness, not just a flag):**
  1. `env.occupancy(..., speed=…, min_speed=…)` requires a **precomputed `speed` array** (`trajectory.py:99,281`) — it does not derive speed from positions. So a `min_speed` knob alone is insufficient; the batch path must obtain a speed signal.
  2. **Filtering occupancy alone gives WRONG rate maps.** Occupancy is the rate denominator; spike counts are the numerator. The spike binner currently counts spikes over the **full** trajectory window (`_binning.py:120,138`) while batch occupancy is computed **separately** (`_binning.py:323`). If `min_speed` masks occupancy but not spikes, numerator and denominator cover different time samples → biased/wrong rates.
- **Change:** add keyword-only **`speed: NDArray | None = None`** and **`min_speed: float | None = None`** to `compute_spatial_rate(s)`. Define an explicit **auto-speed policy** when `speed is None` (e.g. finite-difference of `positions`/`times`, documented), or require the caller to pass `speed`. Build **one shared valid-sample / run mask** from `min_speed` (plus the existing finite/in-window checks) and apply that **same mask to both** the spike-binning and the occupancy computations so numerator and denominator stay aligned. Then `decode_session`/`decode_session_summary` may forward `speed`/`min_speed`.
- **Tests:** masked spikes and masked occupancy use an **identical** sample set (numerator/denominator alignment); `compute_spatial_rates(..., min_speed=v)` parity with a hand-masked reference; passing an explicit `speed` array matches the auto-derived one within tol; default `None`/no-`min_speed` leaves current behavior byte-for-byte unchanged.

**PR deliverable:** Phase 2 PR with a `benchmarks/` (or `tests/.../test_scaling.py`) asserting the memory/throughput bounds on small fixtures; CHANGELOG `### Added`/`### Performance`.
