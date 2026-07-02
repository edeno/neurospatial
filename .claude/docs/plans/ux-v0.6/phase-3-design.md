# Phase 3 — Interop & Ergonomics · Design / RFC

**Status:** Approved 2026-06-05 · **Target:** 0.6.0 (D2 — full scope) · **Gate for:** [`phase-3-interop-and-ergonomics.md`](./phase-3-interop-and-ergonomics.md)

This is the design lock the Phase 3 gate requires. It fixes the three items the
gate names — the **input Protocol surface**, the **`Session` bundle shape**, and
the **`BayesianDecoder` API** — plus the interop/testing strategy and delivery
sequencing. It obeys [`api-contract.md`](./api-contract.md): array-first,
duck-typing/Protocols (never `isinstance` on third-party types), optional deps
stay optional, `io/` performs I/O and returns standard types, result objects and
the `Session` bundle are `frozen`, keyword-only numerical args, raise-don't-print.

## Scope decision

**Full Phase 3 (3.1–3.6) targets 0.6** (D2). Delivered as **6 focused PRs into
`feat/ux-v0.6`**, in dependency order:

1. **3.1** Protocols + `EnvironmentLike` fix (+ pynapple shim lives here but its
   tests are extra-gated)
2. **3.6** `SpikeTrains` container
3. **3.2** `restrict` / epochs
4. **3.5** `Session` bundle + `load_session`
5. **3.4** `BayesianDecoder`
6. **3.3** NWB lazy reads + result round-trip (heaviest, optional-dep) — last

Each PR: implementer → spec-compliance review → code-quality review →
`/pr-review-toolkit:review-pr` → address → merge. Same model as Phases 0–2.

**Hard prerequisite (met):** Phase 1's §1.1 unit-identity threading landed —
results carry `unit_ids`. The pynapple/NWB adapters and `Session.spikes` MUST
carry `unit_ids` into results so interop does not silently re-drop identity.

## Codebase grounding (recon 2026-06-05)

- Net-new: `src/neurospatial/_typing.py`, `recording.py`, `io/pynapple.py`,
  `SpikeTrains`, `decoding/estimator.py` — none exist yet.
- Extras present: `nwb` (`pynwb`, `ndx-pose`, `ndx-events`), `xarray`.
  **No `pynapple` extra** — to be added. Neither `pynapple` nor `pynwb` is in
  the default dev/CI env (only `xarray` is), so interop tests must be
  extra-gated + given dedicated CI jobs (the existing napari/jax/nwb pattern).
- Reusable seams: `encoding/_spikes.py::as_spike_trains` (the spike-train
  boundary adapter), `decoding/posterior.py::SpatialRatesLike` (Protocol pattern
  to mirror), `io/nwb/_units.py::read_units` already returns
  `(list[NDArray], unit_ids)` (a `SpikeTrains` wraps exactly this), and the rich
  `io/nwb/` module set (`_behavior`, `_pose`, `_units`, `_fields`, `_events`, …).
- `EgocentricPolarEnvironment` (`environment/polar.py`) is a **sibling** of
  `Environment`, so `isinstance(polar, Environment)` is `False` — the surprise
  `EnvironmentLike` fixes.

## Governing design principles (locked)

- **Arrays are the universal baseline.** Third-party objects (pynapple/NWB) are
  converted to plain arrays at the *public boundary*; the scientific core never
  sees them and never imports their libraries.
- **Duck-typing via Protocols, never `isinstance` on third-party types.**
- **Optional deps stay optional.** Absence never breaks the array path or import.
- **Immutability.** Result objects, `Session`, and `BayesianDecoder` are frozen;
  "modify" returns a new object.
- **`io/` is pure I/O**; scientific modules never `import pynwb`/`pynapple`.

---

## 3.1 — Protocols + pynapple ingress

**New `src/neurospatial/_typing.py`** — structural (runtime-checkable where
useful) Protocols capturing the *minimal* attributes each consumer needs:

- **`SpikeTrainsLike`** — normalizable to per-unit trains. Accepted inputs:
  `Sequence[NDArray]` (canonical), a 2-D NaN-padded `(n_units, max_spikes)`
  array, the `SpikeTrains` container (§3.6), or a pynapple-`TsGroup`-like object
  (iterable of per-unit timestamp arrays with an `.index`/keys carrying unit
  ids). A single boundary adapter — **extend `encoding/_spikes.py::as_spike_trains`**
  — returns `(list[NDArray], unit_ids)`. `unit_ids` from a `TsGroup`/`SpikeTrains`
  are preserved (prerequisite invariant).
- **`PositionLike`** — `(times, positions)` arrays, or an object exposing `.t`
  (timestamps) + `.values`/`.d` (position samples), e.g. pynapple
  `Tsd`/`TsdFrame`. Boundary adapter → `(times, positions)`.
- **`EnvironmentLike`** (the bonus fix) — the shared surface of `Environment`
  and `EgocentricPolarEnvironment` (`bin_centers`, `n_bins`, `neighbors`,
  `bin_at` where applicable, …). Replace internal `isinstance(env, Environment)`
  checks that misfire on the polar sibling with this Protocol.

**Adapter rule:** conversion happens once, at the public entry (e.g. the first
line of `compute_spatial_rate(s)` / decode entries), producing arrays; everything
downstream is array-only. No `isinstance(x, TsGroup)` anywhere.

**pynapple shim — `neurospatial/io/pynapple.py` only** (scientific modules never
import pynapple):

- `from_pynapple(obj) -> arrays` — `TsGroup` → `(trains, unit_ids)`;
  `Tsd`/`TsdFrame` → `(times, positions)`; `IntervalSet` → `(start, end)` arrays.
- `to_pynapple(result, kind=...) -> Ts…` — e.g. a decoded MAP track → `TsdFrame`.
- `import pynapple` is **lazy inside these functions**; a clear `ImportError`
  (install `neurospatial[pynapple]`) if absent.

**New `pynapple` optional extra** in `pyproject.toml`.

**Tests:** array path byte-for-byte unchanged; the pynapple-flow tests are
`skipif(pynapple absent)` + a dedicated CI job installing the extra; with the
extra absent, import and the array path still work. `EnvironmentLike`: a polar
env now satisfies the Protocol where a bare `isinstance(_, Environment)` failed.

## 3.2 — `restrict` / epochs (array-native, pynapple-compatible)

**New `src/neurospatial/behavior/epochs.py`.**

- `restrict(times, *arrays, epochs) -> (times', *arrays')` — slice each array to
  samples whose `times` fall inside any epoch. `epochs` accepts `(start, end)`
  arrays **or** an `IntervalSet`-like object (duck-typed: exposes `.start`/`.end`
  or is convertible via `from_pynapple`). Normalized internally to array
  intervals; result is a boolean-mask selection (order preserved).
- One-liner: `t, pos = restrict(times, positions, epochs=run_epochs)` then feed
  the restricted arrays to `compute_spatial_rates`.

**Tests:** restrict-by-array-epochs equals restrict-by-`IntervalSet` (extra-gated
for the pynapple side); downstream `compute_spatial_rates` honors the
restriction; empty-epoch and out-of-range handling raise/return-empty cleanly.

## 3.3 — NWB: lazy reads + result round-trip

**Files:** `io/nwb/_behavior.py`, `_pose.py`, `_units.py`, `_fields.py`.

- **`lazy=True`** on position/pose/units reads: return a lazy handle that
  materializes arrays on slice (keeps big recordings off-RAM). Default
  `lazy=False` = current eager behavior, byte-for-byte unchanged.
- **Result round-trip:** add the missing `_fields.py` readers/writers so a
  written `SpatialRatesResult` reads back equal — `read_place_field` reader +
  a units-aligned `write_spatial_rates` with a population `unit` axis.
  `unit_ids` / `unit_table` links preserved.
- I/O stays pure: returns standard types / result objects; never mutates.

**Tests (under the `nwb` extra / CI job):** `write_spatial_rates` →
`read_place_field` → equal, `unit_ids` preserved; `lazy=True` returns without
full materialization (assert peak/handle on a fixture); eager default unchanged.

## 3.4 — `BayesianDecoder` fit/predict — **locks RFC item #3**

**New `src/neurospatial/decoding/estimator.py`.** A thin **immutable** wrapper
over the existing functional core; `decode_position` / `decode_session` /
`decode_position_summary` stay and are the implementation.

```python
@dataclass(frozen=True)
class BayesianDecoder:
    env: Environment
    dt: float = 0.025
    # ... encoding params (bandwidth, smoothing_method, min_occupancy, dtype) ...

    def fit(self, spike_times, times, positions, *, epoch=None) -> "BayesianDecoder":
        # builds encoding models (compute_spatial_rates, fill_value=0.0), optionally
        # restricted to `epoch`; returns a NEW frozen decoder carrying the fitted
        # encoding_models + unit_ids. Does NOT mutate self.
    def predict(self, spike_times, times) -> DecodingResult:         # full posterior
    def predict_summary(self, spike_times, times, *, time_chunk=1024) -> DecodingSummary:  # memory-safe (§2.1)
    def score(self, spike_times, times, positions, *, metric="median_error") -> float:
        # decode + error_against ground truth; metric passthrough to decoding.metrics
```

**Locked decisions:**

- `fit` **returns a new fitted decoder** (frozen; *not* sklearn-style
  self-mutation) — per the immutability constraint. An unfitted decoder's
  `predict`/`score` raise a clear error (mirrors `@check_fitted`).
- `predict` returns a full `DecodingResult`; `predict_summary` returns a
  memory-safe `DecodingSummary` (reuses Phase 2.1). `score` reuses
  `DecodingResult.error_against` / `decoding.metrics`.
- Differentiator vs pynapple `decode_1d/2d`: geodesic/linearized/graph decoding
  through `Environment` (works on `EnvironmentLike`, incl. linearized tracks).
- Inputs are `SpikeTrainsLike`/`PositionLike` (§3.1), so a `TsGroup`/`Tsd` or a
  `Session`'s fields feed it directly.

**Tests:** with identical params and no epoch split, `fit(...).predict(...)`
equals `decode_session` **exactly** (same encoding + same binning + same
full-posterior decode); `predict_summary` MAP == `predict` MAP; `score` matches
`DecodingResult.error_against`; a train/test epoch split works; `fit` does not
mutate the original; unfitted `predict`/`score` raises a clear error.

## 3.5 — `Session` bundle + `load_session` — **locks RFC item #2**

**New `src/neurospatial/recording.py`** — a frozen **discoverability bundle**,
*not* a god-object: heavy compute stays as functions taking the bundle's fields.

```python
@dataclass(frozen=True)
class Session:
    env: Environment | None
    position: PositionLike                 # (times, positions) or a Tsd
    spikes: SpikeTrainsLike                # trains + unit_ids / unit_table
    epochs: "IntervalSetLike | None" = None
    metadata: Mapping[str, Any] | None = None   # optional free-form (e.g. name, subject)

    @classmethod
    def from_arrays(cls, *, env=None, times, positions, spike_times, unit_ids=None,
                    unit_table=None, epochs=None, metadata=None) -> "Session": ...
    @classmethod
    def from_nwb(cls, path_or_file, *, lazy=False, **read_kwargs) -> "Session": ...
    def with_environment(self, env) -> "Session": ...   # returns new
    def restrict(self, epochs) -> "Session": ...          # returns new (uses §3.2)

def load_session(source, **kwargs) -> Session:            # dispatch: nwb path / arrays
```

- **`Session.position` always exposes `.t` (timestamps) and `.values`
  (positions).** When built via `from_arrays`, the arrays are wrapped in a
  minimal internal `Position` holder conforming to `PositionLike`; a pynapple
  `Tsd`/`TsdFrame` already conforms. So `rec.position.t` / `rec.position.values`
  work uniformly regardless of source, and `Session.times`/`Session.positions`
  convenience accessors return the raw arrays.
- **No heavy analysis methods.** Usage stays functional:
  `rates = compute_spatial_rates(rec.env, rec.spikes, rec.position.t, rec.position.values)`.
  Arrays remain trivially extractable from the bundle's fields.
- `metadata` is the only addition to the plan's sketch — a small free-form map
  for name/subject provenance; kept optional and out of the compute path.

**Tests:** immutability (`with_environment`/`restrict` return new, original
unchanged); `from_arrays` and `from_nwb` produce equivalent bundles (nwb side
extra-gated); arrays remain extractable; `spikes` carries `unit_ids` end-to-end.

## 3.6 — `SpikeTrains` — the one justified new container

**New small frozen dataclass** (home: `encoding/_spikes.py` or a new
`spike_trains.py`):

```python
@dataclass(frozen=True)
class SpikeTrains:
    trains: list[NDArray]                    # ragged per-unit spike times
    unit_ids: NDArray                        # defaults to arange(n)
    unit_table: pd.DataFrame | None = None   # region/quality/depth/inclusion
    def __getitem__(self, unit_id): ...      # label access st[unit_id]
    def __iter__(self): ...; def __len__(self): ...
    def filter(self, query: str) -> "SpikeTrains": ...   # e.g. "region=='CA1'"
```

**Justification (constraint A):** the *only* new container Phase 3 adds — ragged
spike times genuinely don't fit a rectangular array; everything else uses
arrays/xarray. Duck-types as `SpikeTrainsLike`. `read_units` can return one.

**Tests:** `read_units` can return a `SpikeTrains`; it flows through
`compute_spatial_rates` (duck-typed) with `unit_ids` preserved; `filter` selects
by `unit_table`; label access + iteration; immutability.

---

## Testing & CI strategy (cross-cutting)

- **Array path is always tested** in the default env and must stay byte-for-byte
  unchanged where the plan says so.
- **`pynapple` interop** → new `pynapple` extra; tests `skipif(absent)`; a
  dedicated CI job installs `neurospatial[pynapple]` and runs them. Mirrors the
  existing napari/jax gating.
- **NWB interop** → existing `nwb` extra + its CI job; round-trip + lazy tests
  run there.
- `mypy` (now blocking) must stay clean, including the new Protocols.
- Docs: add a **session-first *optional* path** while keeping the array path
  canonical; snippet CI covers the array-native examples (pynapple/NWB snippets
  extra-gated or illustrative-skip).

## Out of scope / deferred

- No mutable god-object; no mandatory xarray/NWB/pynapple; no dask.
- `DecodingResult.posterior_entropy` float32-accumulation alignment (a separate,
  deliberately-deferred numeric decision from Phase 2) is untouched here.

## Definition of done (Phase 3)

1. A `TsGroup`/`Tsd` flows into `compute_spatial_rates` and decode via the
   Protocol adapters; with pynapple absent, import and the array path still work.
2. `EnvironmentLike` removes the `Environment`/`EgocentricPolarEnvironment`
   `isinstance`-False surprise at the internal call sites.
3. `restrict(...)` slices by array epochs or an `IntervalSet` identically.
4. A written `SpatialRatesResult` round-trips through NWB (ids preserved);
   `lazy=True` avoids full materialization.
5. `BayesianDecoder(env,...).fit(...).predict(...)` equals `decode_session` on
   the same split; `fit` is non-mutating; `predict_summary` is memory-safe.
6. `Session` is a frozen bundle (`with_environment`/`restrict` return new);
   `from_arrays` ≡ `from_nwb`; arrays extractable; `unit_ids` carried throughout.
7. `SpikeTrains` is the only new container; duck-types as `SpikeTrainsLike`.
8. `unit_ids` survive every interop path (no silent identity re-drop).
