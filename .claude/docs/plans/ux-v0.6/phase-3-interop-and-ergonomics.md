# Phase 3 — Interop & Ergonomics (additive, optional, exploratory)

**Goal:** Meet users where their data already lives (pynapple/NWB), close the result round-trip, and add the thin friendly ergonomics — *without* a hard dependency or a god-object. Raw arrays stay the baseline.

> **Gate:** this is the only phase with real design risk. Before any code, write `phase-3-design.md` (a short brainstorm/RFC) locking: the input Protocol surface, the `Session` bundle shape, and the `BayesianDecoder` API. **Hard prerequisite: Phase 1's unit-identity threading (§1.1) must land first** — the pynapple `TsGroup`→arrays adapter and `Session.spikes` must carry `unit_ids` into results, else interop silently re-drops the identity §1.1 fixes. **D2 = all of Phase 3 targets 0.6** — the RFC gates *design*, not scope. **Escape hatch:** if the RFC finds the combined scope too wide, Phase 3 may split (subset in 0.6, rest in 0.7); this never blocks Phases 0–2. **D3 = the bundle is named `Session`** (immutable, return-new; never a god-object). See [`PLAN.md`](./PLAN.md).

**Governing:** duck-typing/Protocols (no `isinstance`); optional deps; `io/` stays pure I/O; immutability.

---

### 3.1 — Protocols + pynapple ingress (optional, duck-typed)
- **Files:** new `src/neurospatial/_typing.py` (Protocols), boundary normalization in `encoding/_spikes.py` + decoding/event entries; optional `neurospatial/io/pynapple.py` shim.
- **Change:** define `SpikeTrainsLike` / `PositionLike` Protocols. At public boundaries, accept anything matching (incl. pynapple `TsGroup`/`Tsd`/`IntervalSet`) and convert to arrays via a single adapter — **never** `isinstance(x, TsGroup)`. Add `from_pynapple(...)`/`to_pynapple(...)` helpers. pynapple is an **optional** extra; absence never breaks the array path.
- **Bonus:** introduce an `EnvironmentLike` Protocol to fix the `Environment`/`EgocentricPolarEnvironment` `isinstance`-False surprise.
- **Tests:** array path unchanged; with pynapple installed, a `TsGroup` flows into `compute_spatial_rates`; with it absent, import still works and arrays still work.

### 3.2 — Epoch / `restrict` story (array-native, pynapple-compatible)
- **Files:** new `behavior/epochs.py` or extend existing interval handling.
- **Change:** accept epochs as `(start, end)` arrays **or** an `IntervalSet`-like object; provide `restrict(times, *arrays, epochs)` to slice spikes/positions to run/trial epochs before compute. Keeps the "select my running periods" one-liner array-native while transparently accepting pynapple `IntervalSet`.
- **Tests:** restrict by array epochs == restrict by pynapple IntervalSet; downstream `compute_spatial_rates` honors the restriction.

### 3.3 — NWB: lazy reads + result round-trip
- **Files:** `io/nwb/_behavior.py:85`, `io/nwb/_pose.py:84` (eager → optional `lazy=True`), `io/nwb/_fields.py` (population `unit` axis + a `read_place_field` reader; units-aligned `write_spatial_rates`).
- **Change:** `lazy=True` on position/pose/units reads (return lazy handles, materialize on slice); add the missing readers so written results round-trip; preserve `unit_ids`/`unit_table` links. I/O stays pure (returns standard types / result objects).
- **Tests:** write `SpatialRatesResult` → read back → equal (ids preserved); `lazy=True` returns without full materialization (assert on a fixture).

### 3.4 — `BayesianDecoder` fit/predict (behind the RFC)
- **Files:** new `decoding/estimator.py`, over the existing functional core (`decode_position` stays).
- **API (immutable; constraint 2/5):**
  ```python
  dec = BayesianDecoder(env, dt=0.025).fit(spike_times, times, positions, epoch=train_epoch)
  decoded = dec.predict(spike_times, times)             # -> DecodingResult (full posterior)
  summary = dec.predict_summary(spike_times, times)     # -> DecodingSummary (memory-safe, Phase 2.1)
  score = dec.score(spike_times, times, positions, metric="median_error")
  ```
- **Differentiator vs pynapple's `decode_1d/2d`:** geodesic/linearized/graph decoding via `Environment`. `fit` returns a new fitted decoder (no mutation).
- **Tests:** fit/predict == `decode_session` on the same split; `score` matches `DecodingResult.error_against`; train/test epoch split works.

### 3.5 — Thin `Session` bundle + `load_session` (behind the RFC)
- **Files:** new `recording.py` (frozen dataclass), `io/` loaders.
- **API (frozen, return-new — NOT a god-object):**
  ```python
  @dataclass(frozen=True)
  class Session:
      env: Environment | None
      position: PositionLike      # times+positions (or Tsd)
      spikes: SpikeTrainsLike     # trains + unit_ids/table
      epochs: "IntervalSetLike | None" = None
      def with_environment(self, env) -> "Session": ...   # returns new
      def restrict(self, epochs) -> "Session": ...        # returns new
  rec = ns.load_session("rat01.nwb")                # or Session.from_arrays(...)
  rates = compute_spatial_rates(rec.env, rec.spikes, rec.position.t, rec.position.values)
  ```
  A discoverability *bundle*, not a method-bearing universe: heavy compute stays as functions taking the bundle's fields.
- **Tests:** immutability (`with_environment` returns new, original unchanged); `from_arrays` and `from_nwb` produce equivalent bundles; arrays remain extractable.

### 3.6 — Minimal `SpikeTrains` convenience (optional)
- **Files:** new small frozen dataclass `(trains: list[NDArray], unit_table: pd.DataFrame | None)`.
- **Justification (constraint A):** the *only* new container we add, justified solely because ragged spike times don't fit a rectangular array; everything else uses arrays/xarray. Label access `st[unit_id]`, `st.filter("region=='CA1'")`.
- **Tests:** `read_units` can return a `SpikeTrains`; it duck-types as `SpikeTrainsLike`.

**PR deliverable:** RFC (`phase-3-design.md`) → then PRs for 3.1–3.6, **all targeting 0.6 per D2**. Update docs to a session-first *optional* path while keeping the array path canonical.
