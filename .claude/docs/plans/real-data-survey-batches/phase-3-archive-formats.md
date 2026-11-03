# Phase 3 — Breadth: non-NWB archives

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md) · [appendix](appendix.md#phase-3)

Independent of phase 2. Needs phase 1 only for the shared FINDINGS tables.

**Inputs to read first:**

- `datasets/README.md` — "Add a dataset".
- `datasets/load_yartsev_bat_demo.py` — zip members by byte range and MATLAB `table` objects with `mat-io`.
- `datasets/load_rosenberg_labyrinth.py` — a non-NWB loader that builds a track graph.
- `datasets/_fetch.py:90` — `fetch_zip_member`.
- Memory note `matlab-table-reading` (mat-io; Deflate64 needs `inflate64`; remote zip members via fsspec).

**Designs referenced:** [safe pickles](designs.md#safe-pickles) (any pickled array), [MATLAB v7 files](designs.md#matlab-v7-files), [new operation columns](designs.md#new-operation-columns) (`rates.egocentric_boundary`).

## Tasks

- **Open before committing.** For each dataset in the
  [appendix table](appendix.md#phase-3), list the archive (by byte range where
  it is a zip) and confirm it holds per-sample positions and spike times (or,
  for the water maze, positions only). Drop a dataset that holds only
  summaries (e.g. if Zhang 2022's `sdata` has no raw tracking) and record why
  in FINDINGS. Record what was found in the appendix row's place in the
  loader docstring.
- **Expectations first (own commit).** `load_<key>.py` files with `DATASET`
  and `EXPECTATIONS`, registered; commit.
- **Loaders.** Implement each. Format work expected: Axona `.pos` (header then
  binary records; read the header's `pixels_per_metre`) and `.cut` +
  tetrode spike files for Ólafsdóttir — parse by hand with NumPy, cite the
  format document used; Deflate64 for Grieves (`uv add --group datasets
  inflate64`, then check `uv.lock` changed only for it); EthoVision JSON for
  the water maze; MATLAB v5/v7 for the rest. J16's `.npz` has no pickles:
  `np.load(..., allow_pickle=False)`; use its track graph for `env.from_graph`.
- **New operation.** `rates.egocentric_boundary` per
  [designs.md#new-operation-columns](designs.md#new-operation-columns), run on
  every dataset with head direction (old and new).
- **Run and log.** As in phase 2, including the origin-share and
  heading-coverage tables. Grieves is a second 3-D dataset: re-run the
  information-bias null study (`datasets/studies/information_bias_null.py`)
  with it added and update its FINDINGS section.
- **FINDINGS / README.** As in phase 2.

## Deliberately not in this phase

- `src/neurospatial/` changes, including any Axona or MATLAB reader in the
  package: format readers stay in loaders (FINDINGS, "Should loader code move
  into neurospatial?").
- Processed-only deposits (replay-event outputs without spikes) and
  embargoed or restricted records listed in the catalog.

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_formats.py::test_axona_pos_parser_round_trip` | a `.pos` file written by the test from known samples (header + records, per the cited format) parses back to the same positions and timestamps |
| `datasets/tests/test_operations_egocentric.py::test_egocentric_boundary_operation_finds_simulated_cell` | a simulated cell firing when a wall is 20 cm to the animal's left: the operation's peak is within one distance bin of 20 cm and one direction bin of +π/2 |
| `uv run ruff check datasets && uv run ruff format --check datasets` | clean |
| Real run (manual, network, slow) | all loaders validate; results and gap entries committed; no `unreviewed` row |

## Fixtures

Synthesized Axona `.pos` bytes in the test; seeded simulated egocentric
boundary cell in `conftest.py` (walled square arena, random-walk trajectory
with headings).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- User-facing documentation listed as tasks is updated, not deferred.
- The expectations commit precedes every loader implementation commit, and the PR merges with a merge commit.
- No loader trusts a stored unit, scale or sampling rate without a range check recorded in `conversions`.
