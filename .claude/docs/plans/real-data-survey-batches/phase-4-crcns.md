# Phase 4 — Breadth: CRCNS hippocampal sets

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md#phase-4)

Independent of phases 2–3. The hc-11 cross-check needs phase 1's
`grosmark_000044_linear` loader.

**Inputs to read first:**

- `datasets/README.md` — "Add a dataset".
- `datasets/load_grosmark_000044_linear.py` (from phase 1) — the same session as hc-11 Achilles_10252013.
- Memory note `crcns-credentials`: `CRCNS_USERNAME` / `CRCNS_PASSWORD` from `~/.zshrc`, used by name, never printed or written to a file, a log or a result.

## Tasks

- **Access.** Find how CRCNS serves files to an authenticated script (the
  NERSC download portal). Use the credentials only through the environment
  variables. If the site requires a browser step or bot check, stop: do not
  work around it; record the set as out of reach in FINDINGS. Cache
  downloaded files under `data/datasets/` like every other loader and pin
  their SHA-256 after the first download.
- **Expectations first (own commit).** One `load_<key>.py` per row of the
  [appendix table](appendix.md#phase-4) with `DATASET` and `EXPECTATIONS`;
  register; commit. Choose sessions from each set's documentation, smallest
  first.
- **Loaders.** Neuroscope `.res`/`.clu`/`.whl` for hc-3 and hc-2 (positions
  in pixels or cm per the docs; two LEDs give head direction: compute it and
  state the frame), MATLAB for hc-11, hc-13, hc-28/hc-6, hc-31. hc-13's
  `rawpos` zeros: the loader sets untracked frames to NaN and adds a hazard
  that restores the zeros, which is the origin-share check's real-file test.
- **Cross-check.** `hc11_achilles` vs `grosmark_000044_linear`: same spikes
  (count per unit, times within one sample), same positions after unit
  conversion, same epochs. Record each difference as a gap or data quirk.
- **Run and log.** As in phase 2, including the origin-share and
  heading-coverage tables.
- **FINDINGS / README.** As in phase 2; README notes that CRCNS loaders need
  an account and which environment variables they read.

## Deliberately not in this phase

- `src/neurospatial/` changes, including a Neuroscope reader in the package.
- CRCNS sets without sorted units (hc-19) or with precomputed maps only
  (hc-22).
- Storing credentials anywhere but the user's environment.

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_formats.py::test_whl_reader_marks_untracked_as_nan` | a `.whl` text written by the test with `-1` rows for untracked samples gives NaN there and the given values elsewhere |
| `datasets/tests/test_formats.py::test_two_led_heading_frame` | two LED tracks with the front LED due east of the back LED give heading 0; due north (y up) gives +π/2 |
| `datasets/tests/test_fetch.py::test_crcns_fetch_never_logs_credentials` | with dummy values in `CRCNS_USERNAME`/`CRCNS_PASSWORD` and the HTTP call stubbed, neither value appears in captured stdout, stderr, logs or the raised error message |
| `uv run ruff check datasets && uv run ruff format --check datasets` | clean |
| Real run (manual, network, slow) | loaders validate; cross-check table committed; no `unreviewed` row |

## Fixtures

Synthesized `.whl` text and LED tracks in the tests; dummy credentials set
with `monkeypatch.setenv`.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- User-facing documentation listed as tasks is updated, not deferred.
- No credential value appears in code, results, logs or commit history.
- The expectations commit precedes every loader implementation commit, and the PR merges with a merge commit.
