# Phase 3 — Imaging input and NWB files with misplaced or mislabelled data

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md)

Prerequisite: phase 2 merged.

**Inputs to read first:**

- `datasets/` as merged — especially `operations.py`, `GAPS.md`, `results/matrix.md`.
- `neurospatial.io.nwb` readers: `read_position`, `read_head_direction`, `read_units` (`tests/data/public_api.txt:322-335`) and their source in `src/neurospatial/io/nwb/` (how they locate data, and what they do with `unit`, `conversion`, `rate`).
- `docs/user-guide/interoperability.md` — documented NWB reading recipes and the existing note on DANDI 000447 (non-spatial values in a position series, ~line 715).
- `neurospatial.encoding.compute_tuning_curve(s)` and any documented path for continuous (non-spike) signals — search `docs/user-guide/` for "calcium", "fluorescence", "ΔF/F", "deconvolved" before concluding there is none.
- Dataset specifics: [miniscope_002030](appendix.md#miniscope_002030-phase-3), [giocomo_000405](appendix.md#giocomo_000405-phase-3), [giocomo_001701](appendix.md#giocomo_001701-phase-3).
- Memory note `nwb-nonstandard-locations` (walk the whole file before calling data missing).

**Contracts referenced:** [Recording](shared-contracts.md#recording) (incl. `activity`, `activity_kind`, `params["nwb_path"]`), [Operation and Outcome](shared-contracts.md#operation-and-outcome), [Gap-log entry](shared-contracts.md#gap-log-entry) — do not weaken.

**Designs referenced:** [loader recipe](designs.md#loader-recipe) (step 3–4: readers first, trust ranges over labels).

## Tasks

- **Expectations first (own commit).** Add `datasets/load_miniscope_002030.py`, `load_giocomo_000405.py`, `load_giocomo_001701.py` with `DATASET` + `EXPECTATIONS` only; register in `run.py:LOADERS`; commit.
- **Loaders.**
  - `miniscope_002030`: `positions` (cm), `headings` (rad, after checking frame), `activity` = ΔF/F (n_frames, n_cells), `activity_times`, `activity_kind="dff"`; `spike_times=None`.
  - `giocomo_000405`: read `acquisition/track_data` by column name from its description string (`x,y,hd,sp,ha`), convert the mislabelled "mm" values to cm only after confirming the ranges (record the evidence), `headings` from `hd` wrapped to [−π, π]; `activity` = `acquisition/firing_rates`, `activity_kind="binned_rate"`, `activity_times` from the series (note that its `rate` attribute holds the period).
  - `giocomo_001701`: `positions` cm, `headings` stored as "degrees" but radians (confirm from ranges), units with spike times.
  - All NWB loaders (including phase 1–2 ones) set `params["nwb_path"]` to the cached file.
- **New operations:**
  - `io.nwb_readers` (requires `params["nwb_path"]`): call `read_position`, `read_head_direction`, `read_units` on the raw file with default arguments; for each, record `ok` if the values match the loader's range-checked arrays (after the loader's documented conversions), otherwise the specific mismatch (wrong units, not found, silently wrong). The overall status is `workaround` with a gap id when any reader needed non-default arguments or failed, `error` if one raised. A reader that returns degrees-labelled radians multiplied by π/180 without warning is a `wrong-result` gap.
  - `imaging.rates` (requires `activity`, `positions`): look for a public path that turns a continuous signal into a spatial map. If one exists, use it (`ok`). If not, return `missing` with a gap, and add a separate op `imaging.rates_workaround` that converts activity to event times by a stated rule (for `deconvolved`: frames with activity > 0, weighted by amplitude if the API supports weights; for `dff`: threshold at a stated multiple of the robust noise SD, cite where the rule comes from), runs `compute_spatial_rates`, and returns `workaround`. For `binned_rate` data, the workaround is an occupancy-weighted mean of the rates per spatial bin computed in the harness (`env.bin_at` + `np.bincount`), clearly labelled.
- **Dataset hazards** ([contract](shared-contracts.md#loader-module), [design](designs.md#warning-calibration)). These datasets are where hazards come from real files: `giocomo_000405` — `positions_read_as_mm` (a user trusting the "mm" label divides by 10); its `hd` column turned out to be movement direction, so it has no heading hazard; `giocomo_001701` — `headings_read_as_degrees` (`np.radians` applied to values already in radians, as a reader trusting the label would); `miniscope_002030` — any frame/clock or unit conversion the loader records.
- **Run and log.** Run the driver on all eight datasets; commit results; give every non-`ok` cell a gap entry and every new `results/warnings.md` row a verdict. Re-run `io.nwb_readers` on the phase 1–2 NWB datasets and log anything new.
- **README.** Add the three datasets; document `nwb_path`.

## Deliberately not in this phase

- Changes under `src/neurospatial/` — including a fix to a reader that misreads units. Log it with evidence; it ships after phase 4.
- Adding a calcium/ΔF/F input path to neurospatial, or choosing a deconvolution method — the phase only shows what is missing and what the workaround costs.
- 000054 / 001361 (VR imaging with the −500 sentinel) — candidate for the next batch; phase 4 decides.

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_operations_nwb.py::test_readers_flag_mislabelled_heading_unit` | a small NWB written in `tmp_path` with pynwb, `CompassDirection` labelled "degrees" holding radians, makes `io.nwb_readers` report a heading mismatch (not `ok`) |
| `test_operations_nwb.py::test_readers_ok_on_well_formed_file` | the same file with the correct "radians" label gives `ok` |
| `test_operations_nwb.py::test_readers_find_position_in_generic_timeseries` | a `TimeSeries` in `acquisition` with columns described in its description is reported as "not found by read_position" (documents current behavior; the expected status is a finding, not a pass/fail on neurospatial) |
| `datasets/tests/test_operations_imaging.py::test_binned_rate_workaround_recovers_field` | simulated binned rates from a known place field: the harness workaround's peak bin is within one bin of the true field centre |
| `test_operations_imaging.py::test_event_threshold_rule_is_stated` | the event workaround's note includes the threshold rule and its parameters |
| `uv run ruff check datasets && uv run ruff format --check datasets` | clean |
| Real run (manual, network) | three loaders validate; results and gap entries committed; no `unreviewed` row in `results/warnings.md` |

## Fixtures

- Tiny NWB files written with pynwb in `tmp_path` (position, compass direction, units) for reader tests.
- Simulated binned rates and ΔF/F from a known field (seeded) in `conftest.py`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- No loader trusts a stored `unit`, `conversion` or `rate` attribute without a range check recorded in `conversions`.
