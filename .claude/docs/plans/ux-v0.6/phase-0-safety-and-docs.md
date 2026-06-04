# Phase 0 — Safety & Docs (urgent, additive, back-compatible)

**Goal:** Stop silent wrong-science footguns and the broken first-path, and add the one missing golden-path wrapper. Nothing removed or renamed. One PR. This is the highest-value, lowest-risk work and ships first.

**Acceptance for the phase:** all tasks have fail-before/pass-after tests; `ruff`/`mypy` clean; CHANGELOG `### Fixed`/`### Added`; CI green incl. doc-snippets + Windows.

---

### 0.1 — Warn (don't silently drop) spikes outside the trajectory window / inactive bins  ⚑ critical
- **Files:** `src/neurospatial/encoding/_binning.py:120-141` (and the occupancy path it shares).
- **Now:** spikes outside `[times.min(), times.max()]` and spikes mapping to bin `-1` are dropped with no signal (`:121`, `:124`, `:138-141`). A ms-vs-s unit mix → near-empty field, no warning.
- **Change:** after masking, if the dropped fraction exceeds a threshold (default **0.5**, also fire when *all* spikes drop but ≥1 spike existed), `warnings.warn` naming counts, both time ranges, and the units hypothesis: e.g. `"<n>/<N> spike_times (X%) fell outside the position time window [t0, t1]; spike_times.min()=… max()=…. Check that spike_times and times share units (both seconds). Dropped spikes do not contribute."` Same for inactive-bin drops (coordinate-frame/units hint). Add a `warn_on_drop: bool = True` keyword to silence intentionally.
- **Decision:** warn, do **not** raise — a genuinely silent cell is legitimate. (No permissive *silence*, but no false error either.)
- **Tests:** ms-vs-s mismatch → warns + returns near-zero (fail-before: silent); fully-in-window → no warning; `warn_on_drop=False` → silent.

### 0.2 — Normalize array-likes at public boundaries before `.ndim`
- **Files:** `src/neurospatial/behavior/trajectory.py:132,289,591`; audit other public entries (encoding/decoding already do this).
- **Change:** `positions = np.asarray(positions, dtype=float)` inside a typed try/except (pattern from `environment/factories.py:256-269`) **before** any `.ndim`/`.shape` access.
- **Tests:** Python-list input → success or domain `ValueError`, never `AttributeError: 'list' object has no attribute 'ndim'`.

### 0.3 — Symmetric `Inf` validation on `event_times`
- **Files:** `src/neurospatial/events/alignment.py:156` (docstring promise at `:99`).
- **Change:** add `np.isinf(event_times)` rejection mirroring the spike-time check (`:149`).
- **Tests:** `Inf` event time → `ValueError` (fail-before: silent empty result).

### 0.4 — Beginner-grade error for bare `Environment()`
- **Files:** `src/neurospatial/environment/core.py:335`.
- **Change:** replace `ValueError("layout parameter is required")` with the factory-guidance message already used at `environment/decorators.py:97-108` (point to `Environment.from_samples(...)`).
- **Tests:** `Environment()` error message contains "from_samples".

### 0.5 — Remove internal-doc references from user-facing errors
- **Files:** `src/neurospatial/layout/validation.py:334,400`.
- **Change:** drop "See CLAUDE.md section …"; replace with the public docs URL or "please report at <github issues>".
- **Tests:** no user-facing error string contains "CLAUDE.md" (add a repo-wide grep test).

### 0.6 — Docs sweep (one canonical beginner path)
- **Files:** `docs/user-guide/workflows.md` (Workflow 1 rewrite to the `compute_spatial_rate` idiom; remove `np.histogram`/`grid_shape`/`gaussian_filter` reimplementation `:44-101`); kill `compute_firing_rate` refs (`workflows.md:304,312`, `docs/user-guide/spatial-analysis.md:871`); `env.plot(field,…)` → `env.plot_field(field,…)` (11 sites in `rl-primitives.md`, `spike-field-primitives.md`); bump `v0.4.0`→`v0.5.0` (`README.md:85,660`, `docs/index.md:97,109`); fix `examples/20_bayesian_decoding` to use `compute_spatial_rates(...).firing_rates` + `bin_spikes_in_time` (via **jupytext**, edit `.py` then `--sync`).
- **Tests:** add the rewritten workflow snippets to the doc-snippet harness (`scripts/test_doc_snippets.py` / `docs/snippets.yml`); CI doctest/snippet job covers them.

### 0.7 — `decode_session()` golden-path wrapper  ⚑ headline
- **Files:** new function in `src/neurospatial/decoding/` (export from `neurospatial.decoding`).
- **API:**
  ```python
  def decode_session(env, spike_times, times, positions, *,
                     dt=0.025, bandwidth=5.0, smoothing_method="diffusion_kde",
                     min_occupancy=0.0, encoding_models=None,
                     **decode_kwargs) -> DecodingResult:
      """Encode → bin → decode in one call. Glue over compute_spatial_rates,
      bin_spikes_in_time, decode_position. encoding_models optional (else fit here)."""
  ```
- **Implementation (verified signatures):**
  ```python
  trains = normalize_spike_times(spike_times)          # -> list[NDArray], one per neuron
  times_arr = np.asarray(times, dtype=np.float64)      # accept list/array; normalize once (§0.2)
  if encoding_models is None:
      encoding_models = compute_spatial_rates(
          env, trains, times_arr, positions,
          bandwidth=bandwidth, smoothing_method=smoothing_method,
          min_occupancy=min_occupancy, fill_value=0.0,
      ).firing_rates                                   # (n_neurons, n_bins)
  counts, centers = bin_spikes_in_time(                # expects a SEQUENCE of per-neuron arrays
      trains, dt, t_start=times_arr.min(), t_stop=times_arr.max())
  return decode_position(env, counts, encoding_models, dt, times=centers, **decode_kwargs)
  ```
  Returns `DecodingResult` (stable type — constraint 2).
- **Note A — normalize once (finding):** `bin_spikes_in_time` expects a **sequence of per-neuron arrays**; a bare 1-D single-neuron array passed directly is iterated as scalar "trains" (`decoding/_binning.py:141`) → wrong neuron axis / later shape mismatch. So **normalize `spike_times` once** with the same normalizer `compute_spatial_rates` uses (`encoding/_spikes.py:normalize_spike_times`, currently **private**) and feed that list to **both** the encoder and `bin_spikes_in_time`. Since `decode_session` lives in `decoding/`, **promote `normalize_spike_times` to a shared/public util** (or re-export it) rather than duplicating the logic or cross-importing a private symbol.
- **Note B — signatures (finding #2):** `bin_spikes_in_time` is `(spike_trains, dt, t_start=None, t_stop=None, *, orient="time_x_neuron")` → `(counts, centers)`; pass `dt` **positionally**, the trajectory window via `t_start/t_stop`. `compute_spatial_rates` has **no `min_speed`** (only `min_occupancy`); speed-filtered batch occupancy is a separate enhancement tracked as **Phase 2 §2.6** — do not invent a `min_speed` forward here until §2.6 lands.
- **Tests:** 5-line golden-path test produces a `DecodingResult` whose MAP tracks a simulated trajectory; equivalence to the manual 3-call path.

### 0.8 — (cheap consistency) `distance_to` missing-region → `RegionNotFoundError`
- **Files:** `src/neurospatial/environment/queries.py:528`.
- **Change:** raise `RegionNotFoundError` (from `_exceptions.py`, keeps `KeyError` base) instead of bare `KeyError`, preserving the helpful available-regions message.
- **Tests:** missing region raises `RegionNotFoundError` and is still caught by `except KeyError`.

**PR deliverable:** `feat/ux-v0.6` ← Phase 0; CHANGELOG; reviewed before Phase 1 begins.
