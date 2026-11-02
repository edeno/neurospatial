# Phase 4 — Gap synthesis and the next batch

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Prerequisite: phase 3 merged. This phase writes no code.

**Inputs to read first:**

- `datasets/GAPS.md`, `datasets/results/matrix.md`, `datasets/results/warnings.md`, `datasets/verdicts.json`, `datasets/results/reference_checks.md`, `datasets/results/*.json` (expectation outcomes, runtime, memory).
- The loaders themselves — every `conversions` entry and every harness workaround is evidence.
- `docs/vision.md` and the design principles in `docs/contributing.md#design-principles` — proposals are judged against them.
- Survey catalog (local, git-ignored): `docs/research/reports/Open spatial neuroscience datasets.md` — for choosing the next batch.

**Contracts referenced:** [Gap-log entry](shared-contracts.md#gap-log-entry).

## Tasks

- **Consolidate.** Merge duplicate gap entries; for each, fill in `Datasets` with every dataset it affected. Mark `data-quirk` entries that neurospatial should *detect or refuse* (e.g. a heading labelled degrees with values in [−π, π]) separately from quirks it should not care about.
- **Write `datasets/FINDINGS.md`.** One section per proposed change, ordered by (scientific risk, number of datasets affected). Each section: the problem in one sentence; evidence (gap ids, datasets, numbers); proposed change at the API level (function/argument/error message), or "needs a design decision" with the options; the module it touches; whether it changes results for existing users. Group into: wrong results, missing capabilities, friction (worked but needed adaptation), error messages, docs, performance.
- **Warning calibration.** From `results/warnings.md` and `datasets/verdicts.json`, write a `FINDINGS.md` section with two lists. *Too aggressive:* each warning with a `false alarm`, `noisy` or `unactionable` verdict — how many datasets it fired on, how many of those were false alarms, and the proposed change (raise the threshold to a stated value with the per-dataset numbers that justify it, emit once per call, or reword the message). *Not aggressive enough:* each `silent` hazard — what the user gets instead (the clean vs hazardous summaries), and the proposed check (warn or raise, and the rule on values that would catch it without firing on any clean dataset in the table). A proposed new warning must show zero hits on the clean runs; a proposed threshold change must show it still catches every hazard it caught before.
- **Reference-check verdicts.** For each cross-tool comparison: agree / disagree-by-convention (documented, no change) / disagree-needs-fix.
- **Generic-loader question.** From the loaders, count how much code is generic (fetch/verify, NWB tree walk, unit range checks, MATLAB tables) vs dataset-specific. Recommend whether any of it belongs in neurospatial (e.g. a `neurospatial.io` helper or a fetcher module) — answer the overview's open question 2 with evidence.
- **Next batch.** Propose the next 4–6 datasets by the empty or weak cells of the matrix (operations never exercised, data shapes never seen). Candidates from the catalog: Grieves 2020 3-D lattice (CC0; Deflate64 zip), CRCNS hc-11 MATLAB or DANDI 000044 (already used by the tutorial), Jadhav W-track with prefrontal (000447/001539), Zhang 2022 egocentric boundary cells (Figshare), VR imaging with sentinels (000054), Gardner 2022 grid modules (CC0; needs `allow_pickle` → decide whether acceptable), the water-maze JSON.
- **Tutorial candidates.** Name the 1–2 loaders that would make the best "use your own recording" examples for `examples/` or `docs/user-guide/`.
- **Ask the maintainer.** End `FINDINGS.md` with the decisions needed (which proposals to pursue, in what order) — do not start any `src/` change from this phase.

## Deliberately not in this phase

- Any change under `src/neurospatial/` or new loaders.
- Filing GitHub issues (the maintainer decides which findings become issues).

## Validation slice

| Test | Asserts |
| --- | --- |
| Every proposal in `FINDINGS.md` | cites at least one gap id; cites ≥2 datasets or one reference-check disagreement, or says explicitly why one dataset is enough |
| Every `open` gap in `GAPS.md` | appears in some `FINDINGS.md` section or is marked `won't fix` with a reason |
| Next-batch list | each dataset names the matrix cell(s) it fills |
| Warning calibration section | every `false alarm`, `noisy`, `unactionable` and `silent` verdict appears; each proposed warning or threshold reports its hits on clean runs and on hazard probes |

## Fixtures

None.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
- Claims in `FINDINGS.md` match the numbers in `results/` (spot-check three).

This phase is documentation only; per maintainer practice, a docs-only change may
go straight to `main` after local checks instead of a PR — ask first.
