# Phase 1 — Harness, two reference datasets, and cross-tool checks

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

**Inputs to read first:**

- [data/load_bandit_data.py](../../../../data/load_bandit_data.py) — existing download/verify pattern that `_fetch.py` generalizes (lines 1-60).
- [pyproject.toml:131](../../../../pyproject.toml) (`testpaths`), `:175-232` (ruff), `:446` (`[dependency-groups]`).
- `src/neurospatial/environment/core.py:3288` (`from_samples`), `:1903` (`occupancy`).
- `src/neurospatial/encoding/spatial.py:4129` (`compute_spatial_rates`), `:1940` (`SpatialRatesResult`), `:2380` (`spatial_information`).
- `src/neurospatial/encoding/grid.py:180` (`spatial_autocorrelation`), `:813` (`grid_score`); `src/neurospatial/encoding/border.py:37` (`border_score`).
- `src/neurospatial/encoding/directional.py:2585` (`compute_directional_rates`).
- `src/neurospatial/decoding/estimator.py:95,647,866` (`BayesianDecoder`, `fit`, `predict`); `src/neurospatial/decoding/_result.py:337` (`DecodingResult`); `src/neurospatial/decoding/metrics.py:290` (`median_decoding_error`).
- `src/neurospatial/io/nwb/` readers and writers (names in `tests/data/public_api.txt:322-341`).
- pynapple tutorials at the pinned version: `doc/examples/tutorial_null_distributions.md`, `doc/examples/tutorial_HD_dataset.md`.
- `.claude/rules/executable-docs.md` only if you touch docs (this phase should not).

**Contracts referenced:**

- [Recording](shared-contracts.md#recording), [Loader module](shared-contracts.md#loader-module), [Operation and Outcome](shared-contracts.md#operation-and-outcome), [Gap-log entry](shared-contracts.md#gap-log-entry), [Folder layout](shared-contracts.md#folder-layout) — implement exactly; do not weaken.

**Designs referenced:** [fetch](designs.md#fetch-and-cache), [validation](designs.md#record-validation), [expectations](designs.md#expectations), [registry](designs.md#operation-registry), [driver](designs.md#driver), [warning calibration](designs.md#warning-calibration), [reference checks](designs.md#reference-checks), [loader recipe](designs.md#loader-recipe).

Dataset specifics: [appendix: sargolini_000582](appendix.md#sargolini_000582-phase-1), [appendix: peyrache_mouse32](appendix.md#peyrache_mouse32-phase-1).

## Tasks

- **Tooling.** Add the `datasets` dependency group and the `"datasets/**/*.py" = ["T201"]` ruff per-file ignore to `pyproject.toml` (see [overview: dependency policy](overview.md#dependency-policy)); run `uv lock`; commit with `uv.lock`.
- **Harness core.** Create `datasets/_fetch.py`, `datasets/_record.py` (`Recording`, `validate_recording`, `MEASURES`, `check_expectations`), `datasets/operations.py` (phase-1 registry incl. `nwb_roundtrip`), `datasets/_calibration.py` (`captured`, `GENERIC_HAZARDS`, `probe_hazards`), `datasets/run.py` (driver + `results/matrix.md` and `results/warnings.md` rendering), as designed. No `__init__.py` in `datasets/`.
- **Offline tests.** `datasets/tests/conftest.py` puts `datasets/` on `sys.path` and builds a simulated `Recording` with `neurospatial.simulation` (place cells in a 100 cm box, seeded RNG; read `src/neurospatial/simulation/` for the current generator names). Tests per the validation slice below.
- **Pre-registered expectations (own commit, before any real run).** Create `datasets/load_sargolini_000582.py` and `datasets/load_peyrache_mouse32.py` containing only `DATASET`, `EXPECTATIONS` (seeded from the appendix; refine from the paper/readme, never from running the loader) and a `load` that raises `NotImplementedError`. Commit.
- **Loaders.** Implement both `load` functions per the [loader recipe](designs.md#loader-recipe). Try neurospatial's NWB readers first and log every failure or surprise as a gap. Choose `params` with a justification comment each (e.g. `bin_size` 2.5 cm for a 100 cm box: 40 bins, matching the pynapple reference; `min_speed` 2.5 cm/s if used, citing the convention; `train_epochs`/`test_epochs` as alternating 60 s blocks of the session for held-out decoding).
- **Dataset hazards.** In each loader, add `HAZARDS` for any dataset-specific quirk the loader corrects (none is expected for these two beyond the generic hazards; if a loader records a correcting conversion, it gets a hazard). Write them after the loader works, not with the expectations.
- **Gap log.** Create `datasets/GAPS.md` with the entry format from the contract; add entries as they arise (loader friction, reader failures, data quirks, error-message quality).
- **Run.** `MPLBACKEND=Agg uv run --group datasets --extra nwb --extra pynapple python datasets/run.py`; commit `results/*.json` and `results/matrix.md`. Review every `error`/`workaround`/`missing` cell and give it a gap entry.
- **Warning verdicts.** Create `datasets/warning_verdicts.json` and give every row of `results/warnings.md` a verdict, with a gap entry where the [verdict table](designs.md#warning-calibration) calls for one. Open the triggering samples before calling a warning a false alarm. Commit the regenerated `results/warnings.md` with no `unreviewed` rows.
- **Reference checks.** Implement `datasets/reference_checks.py` (all three checks in [designs](designs.md#reference-checks)); run; commit `results/reference_checks.md`. Every disagreement beyond the investigation trigger gets a gap entry with the convention analysis.
- **README.** `datasets/README.md`: purpose (one paragraph), how to run, cache location and `NEUROSPATIAL_DATA_DIR`, licenses (data are not redistributed), how to add a dataset (point to the loader contract), how to read the matrix.

## Deliberately not in this phase

- Any change under `src/neurospatial/` — even an obvious one-line fix found here. Log it; fixes ship separately after phase 4.
- 1-D tracks, graphs, replay, 3-D, imaging operations — phases 2 and 3 add their datasets and operations.
- `decode.heading` beyond what the HD reference check needs; if neurospatial lacks a heading decoder, that is a gap entry, not a harness feature.
- Promoting loaders to `examples/` or docs.
- CI jobs for the harness.

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_record.py::test_valid_simulated_recording_passes` | simulated recording passes `validate_recording` |
| `test_record.py::test_validation_lists_every_problem` | a recording with unsorted spikes, decreasing times and headings in degrees raises one `ValueError` naming all three |
| `test_record.py::test_expectations_report_failures_without_raising` | a wrong `n_units` and an out-of-range `sampling_rate_hz` come back `passed=False` with detail text; unknown key raises `KeyError` |
| `datasets/tests/test_fetch.py::test_fetch_verifies_and_caches` | `fetch` from a local `file://` URL installs once; a second call does not re-read the URL (monkeypatch `urlopen` to fail) |
| `test_fetch.py::test_fetch_refuses_mismatched_existing_file` | an existing file with a different digest raises `FileExistsError` and is left untouched |
| `test_fetch.py::test_bad_download_leaves_no_temp_file` | a digest mismatch raises `ValueError` and leaves no file in the destination directory |
| `datasets/tests/test_run.py::test_driver_statuses_on_simulated_recording` | running the registry on the simulated recording yields `ok` for 2-D ops, `n/a` for `hd.tuning` when headings are `None`, `error` with `"<Type>: <msg>"` when an op raises (inject a failing op) |
| `test_run.py::test_simulated_place_cells_recovered` | on the simulated recording, `rates.spatial_information` median > 0.5 bits/spike and `decode.position` median error < 10 cm (scientific sanity, ground truth known) |
| `test_run.py::test_matrix_render` | `matrix.md` has one row per JSON and the status codes in the right cells |
| `datasets/tests/test_calibration.py::test_captured_groups_by_template` | an injected op that warns `"3 bins"` and `"5 bins"` and then raises yields one template `"# bins"` with count 2 and the exception returned, not raised |
| `test_calibration.py::test_hazard_responses_classified` | on the simulated recording, injected probe ops that raise, warn, and do nothing are reported `raised`, `warned`, `silent`, with clean and hazardous summaries attached |
| `test_calibration.py::test_generic_hazards_apply_only_when_relevant` | a recording without headings or without NaN positions gets no `headings_in_degrees` / `missing_positions_as_zero` probe; a loader `HAZARDS` entry is added to the generic ones |
| `test_calibration.py::test_unreviewed_rows_rendered` | `warnings.md` marks a template missing from `warning_verdicts.json` as `unreviewed` and joins verdicts for the rest |
| `uv run ruff check datasets && uv run ruff format --check datasets` | clean |
| Real run (manual, network) | both datasets load and validate; `results/matrix.md`, `results/warnings.md` and `results/reference_checks.md` committed; every non-`ok` cell has a gap id; no `unreviewed` warning or hazard row |

Run the offline tests with `uv run --group datasets pytest datasets/tests -n 0`
(they're outside `testpaths`). Simulated tests must not need the network.

## Fixtures

- Simulated recording from `neurospatial.simulation`, seeded, built in `datasets/tests/conftest.py` (session-scoped).
- A tiny local file for fetch tests created in `tmp_path`, served via `file://` URL.
- Real data comes only from the cache; no data files are committed.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Expectations were committed before the first real run (check `git log -- datasets/load_*.py`).
- Every loader conversion appears in `Recording.conversions` and, if it compensates for a neurospatial gap or a data quirk, in `GAPS.md`.
