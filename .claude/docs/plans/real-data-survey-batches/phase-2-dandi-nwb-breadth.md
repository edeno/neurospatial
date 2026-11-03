# Phase 2 — Breadth: small DANDI and EBRAINS NWB recordings

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#new-operation-columns) · [appendix](appendix.md#phase-2)

Prerequisite: phase 1 merged (`open_nwb`, `nwb_source`).

**Inputs to read first:**

- `datasets/README.md` — "Add a dataset".
- `datasets/load_giocomo_000405.py`, `datasets/load_giocomo_001701.py` — loaders that trust value ranges over labels.
- `datasets/operations.py:487-578` (`nwb_readers`) and `:682` (`OPERATIONS`).
- `src/neurospatial/io/nwb/_pose.py:42` — `read_pose`, for the pose operation.
- Memory note `nwb-nonstandard-locations`: walk the whole file before calling data missing.

**Designs referenced:** [streamed NWB](designs.md#streamed-nwb) (for any file over about 1 GB), [new operation columns](designs.md#new-operation-columns).

## Tasks

- **Choose files.** For each dataset in the [appendix table](appendix.md#phase-2),
  list the dandiset's assets (DANDI API, sorted by size) and pick the smallest
  file with units and position (for 001845, pose; for Kanter, objects if any).
  Record asset id, version, size and SHA-256 in the loader. Drop a dataset
  whose smallest usable file is above about 15 GB and cannot be streamed, and
  say so in FINDINGS.
- **Expectations first (own commit).** Nine `load_<key>.py` files with
  `DATASET` and `EXPECTATIONS`, registered in `run.py:LOADERS`; commit.
- **Loaders.** Implement each `load()`. Points the batch is meant to exercise:
  `conversion` (001695), blank units (000987), pixels (001754, 000003: keep
  pixels only if no calibration exists, and record it), a file sampled at
  spike times (001754: irregular `times`), NaN-heavy pose (001845: choose one
  keypoint as `positions` and say why), object positions (Kanter). Every
  quirk the loader corrects gets a `HAZARDS` entry.
- **New operations.** `io.read_pose` and, if Kanter has object positions,
  `rates.object_vector`, per [designs.md#new-operation-columns](designs.md#new-operation-columns).
  Datasets without the input get `–` as today.
- **Run and log.** Run `run.py` on all datasets (old and new) on current
  `main`; commit results; gap entries and verdicts as usual. Re-run the
  origin-share and heading-coverage tables from phase 1 with the new
  recordings added (the tables are in FINDINGS).
- **FINDINGS.** Update counts, the probe table, "At a glance", and add new
  proposals only where a gap recurs or changes a result.
- **README.** Add the datasets to the downloads table.

## Deliberately not in this phase

- `src/neurospatial/` changes, including the NWB reader gaps this batch will
  find: log them (they add evidence for the separate NWB-reader plan).
- Imaging dandisets (000054, 002030 siblings): continuous-signal rate maps are
  deferred.
- More than one session per dataset.

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_operations_pose.py::test_read_pose_operation_reports_nan_share` | a tiny NWB with an `ndx-pose` `PoseEstimation` of two keypoints (one 30 % NaN) written in `tmp_path` gives an `ok` outcome whose NaN share for that keypoint is 0.3 ± 0.01 |
| `datasets/tests/test_operations_objects.py::test_object_vector_operation_finds_simulated_field` | a simulated cell firing at a fixed distance and direction from one object: the operation's peak bin is within one bin of the simulated distance and direction (only if `rates.object_vector` is added) |
| `uv run ruff check datasets && uv run ruff format --check datasets` | clean |
| Real run (manual, network, slow) | all loaders validate; results and gap entries committed; no `unreviewed` row |

## Fixtures

Tiny NWB files written with pynwb (and `ndx-pose`) in `tmp_path`; seeded
simulated object-vector cell in `conftest.py`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- User-facing documentation listed as tasks is updated, not deferred.
- The expectations commit precedes every loader implementation commit, and the PR merges with a merge commit.
- No loader trusts a stored unit, `conversion` or `rate` without a range check recorded in `conversions`.
