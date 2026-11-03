# Phase 1 — Datasets the pending decisions wait on

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md) · [appendix](appendix.md#phase-1)

**Inputs to read first:**

- `datasets/README.md` — the harness contract ("Add a dataset") and how to run it.
- `datasets/FINDINGS.md` — "Decisions for the maintainer" (what is gated on this phase) and proposals 3–4 and checks 1–3 under "Warning calibration".
- `datasets/_fetch.py:75-113` — `fetch`, `fetch_zip_member`, `dandi_asset_url`.
- `datasets/_record.py:21-75` — `Recording`; `params` keys at `:52`.
- `datasets/operations.py:153-200` (`information_vs_shifts`), `:487-578` (`nwb_readers`), `:682-705` (`OPERATIONS`).
- `datasets/reference_checks.py:220-280` — `grid_score_check`.
- `datasets/load_rosenberg_labyrinth.py:103-133` — the restricted unpickler to model Gardner's on.
- `datasets/load_figure8_000552.py` — the closest existing loader for 000044 (1-D linearized position, sleep, ripples).
- `src/neurospatial/encoding/spatial.py:1555-1640` — `spatial_information_bias` and its derivation.

**Designs referenced:** [streamed NWB](designs.md#streamed-nwb), [safe pickles](designs.md#safe-pickles), [MATLAB v7 files](designs.md#matlab-v7-files), [information-bias null study](designs.md#information-bias-null-study), [gated evidence](designs.md#gated-evidence).

## Tasks

- **Expectations first (own commit).** Add `load_gardner_2022_grid.py`,
  `load_vollan_lt.py`, `load_vollan_of.py`, `load_grosmark_000044_circular.py`,
  `load_grosmark_000044_linear.py`, `load_jadhav_000447.py` with `DATASET` and
  `EXPECTATIONS` only, register them in `run.py:LOADERS` (`:33`), commit.
  Expectations come from the [appendix](appendix.md#phase-1) and the catalog,
  e.g. unit counts (104 / 137 for the 000044 sessions), sampling rates
  (100 Hz Vollan, about 39 Hz 000044, about 33 Hz 000447), track lengths
  (1.6 m linear), and for the grid sets "most module units have grid score
  above 0.3". Choosing the Vollan open-field file may need reading metadata
  of a few files first (allowed: it is not the analysed data).
- **Streamed NWB support.** Add `dandi_stream_url` and `open_nwb` to
  `_fetch.py`, rename `nwb_path` → `nwb_source` at every use, and route every
  open through `open_nwb`, as in [designs.md#streamed-nwb](designs.md#streamed-nwb).
  Update the README sentence on `nwb_path`.
- **Loaders.** Implement the six `load()` functions:
  - Gardner: per [designs.md#safe-pickles](designs.md#safe-pickles); record
    the inspected pickle globals in the docstring; module label per unit as
    `params["module"]`; open-field window from `_sessions.txt`. Units of
    `x`, `y`, `t` from the value ranges (box size in the paper is 1.5 m;
    verify).
  - Vollan: per [designs.md#matlab-v7-files](designs.md#matlab-v7-files);
    m → cm; `hd` unit and zero direction from value ranges and from
    comparing with the direction of movement during running (the 000405
    method, `load_giocomo_000405.py:22-24`); the 5 cm/s speed filter means
    `times` has gaps: pass them through as they are and record it; author
    module labels and gridness into `params` for the gated evidence.
  - 000044: streamed; 2-D position from `<track>Position`, the linearized
    coordinate into `params["linearized"]`; epochs `PRE`/`MAZE`/`POST`;
    `analysis_epochs` = maze epoch; ripple / sleep-state intervals if present
    (walk the file); `replay_dt` as for the figure-8. Circular track: record
    the linearized coordinate's wrap like G-026 if it wraps.
  - 000447: streamed; keep x, y, drop the velocity column, record the
    compound unit string; trials and wells from the file's intervals into
    `epochs` / `events`.
  - Every loader: `HAZARDS` for each quirk it corrects (m → cm off by 100,
    speed-filter gaps if a user interpolates over them, the velocity column
    read as a third position dimension for 000447).
- **Information-bias null study.** `datasets/studies/information_bias_null.py`
  per [designs.md#information-bias-null-study](designs.md#information-bias-null-study);
  writes `results/information_bias_null.{md,json}`. Smoke-test on 000582
  first and record the runtime.
- **Reference checks.** Extend `reference_checks.py` so the grid check runs
  on Gardner and Vollan open field too, plus option (a)
  ([designs.md#gated-evidence](designs.md#gated-evidence)); keep the 000582
  section unchanged.
- **Run and log.** Smoke-test one streamed loader (bytes read, seconds,
  peak memory) before the full run. Run `run.py` on all 14 datasets on
  current `main`; commit results; give every non-`ok` cell a gap entry
  (`GAPS.md`, continuing at G-037) and every new `warnings.md` row a verdict.
- **Gated evidence.** Add the three sections of
  [designs.md#gated-evidence](designs.md#gated-evidence) to
  `datasets/FINDINGS.md`, each ending in a recommendation; update the
  dataset count, probe table and "At a glance". Add open-arena lap results to
  decision 5's evidence without deciding it.
- **README.** Add the six datasets to the downloads table; document streamed
  sources (nothing is downloaded; DANDI's published SHA-256 is pinned) and
  the `studies/` folder.

## Deliberately not in this phase

- Any `src/neurospatial/` change, including option (a) for the grid score:
  it is computed in the harness only, as evidence.
- Breadth datasets (phases 2–4), even if a phase-1 dataset's dandiset has more
  sessions. One session per dataset key.
- Hazard design beyond the quirks these loaders correct.
- Deciding laps in open arenas (decision 5).

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_fetch.py::test_dandi_stream_url_refuses_changed_digest` | with `urllib.request.urlopen` replaced by a stub returning a different digest, `dandi_stream_url` raises naming both digests |
| `test_fetch.py::test_open_nwb_reads_local_and_http` | a tiny NWB written in `tmp_path` opens through `open_nwb` from its path and from a local `http.server` URL, with equal positions |
| `datasets/tests/test_safe_pickle.py::test_pickle_imports_lists_globals_without_running` | a pickle of an object array of float arrays reports only NumPy globals; a pickle whose `__reduce__` calls `os.system` is listed as `os.system` and the call never runs (check with a sentinel file) |
| `test_safe_pickle.py::test_restricted_npz_refuses_unlisted_global` | an `.npz` holding an object array with a `collections.OrderedDict` raises `UnpicklingError` naming it |
| `datasets/tests/test_information_bias_null.py::test_poisson_ratio_near_one_on_dense_occupancy` | on a simulated dense 2-D occupancy, mean SI of 200 untuned Poisson units / mean bias is within 0.8–1.25 (an independent check that the study measures what it says) |
| `test_information_bias_null.py::test_bursts_raise_information_above_poisson` | at the same mean rate, bursty untuned units have higher mean SI than Poisson ones (seeded) |
| `datasets/tests/test_reference_grid.py::test_nan_aware_autocorrelation_matches_full_overlap` | on a map with no NaN, option (a)'s autocorrelogram equals `scipy.signal.correlate`-based Pearson at every lag within 1e-10 |
| `uv run ruff check datasets && uv run ruff format --check datasets` | clean |
| Real run (manual, network, slow) | six loaders validate; `results/` and `GAPS.md` committed; no `unreviewed` row; three gated sections in FINDINGS |

## Fixtures

- Tiny NWB files written with pynwb in `tmp_path` (existing pattern in
  `datasets/tests/conftest.py`).
- Seeded simulated trajectories and untuned spike trains in `conftest.py`
  for the null-study tests.
- Hand-built pickles in the test (never downloaded) for the safe-pickle tests.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): no `nwb_path` left anywhere in `datasets/`.
- User-facing documentation listed as tasks is updated, not deferred.
- The expectations commit precedes every loader implementation commit, and the PR merges with a merge commit.
- No loader trusts a stored unit, `conversion` or `rate` without a range check recorded in `conversions`.
