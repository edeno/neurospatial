# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

The evidence for every item is in `docs/reviews/UX_REVIEW_2026-11-02.md`. That
review marks which findings were reproduced and which were read in code only.
Re-verify the "agent" and "code" items against the current code before
changing anything.

## Current codebase integration points

- `src/neurospatial/_intervals.py:1699` — `warn_excluded_intervals`. Every
  analysis passes its final interval mask through this function; there are 11
  call sites. Phase 1 adds the partial-exclusion warning here. Phase 6 turns
  the all-excluded case into a raise.
- `src/neurospatial/_intervals.py:1158` — `IntervalGates`. Phase 1 adds a
  field so the bin gate's own exclusions can be measured.
- `src/neurospatial/encoding/_spatial_counts.py:68` — `_DROP_WARN_THRESHOLD =
  0.5`. Phase 1 replaces it with the shared threshold.
- `src/neurospatial/_exceptions.py:68` — `_format_error`, the house
  `What / Why: / Fix:` formatter. Every new message uses it.
- `src/neurospatial/decoding/posterior.py:553` — `_prepare_decode_inputs`,
  the shared front half of every decoder. Phase 1 adds the zero-rate and
  `dt` checks.
- `src/neurospatial/encoding/directional.py:2230,2322` —
  `_check_smoothing_bandwidth` and its call site. Phase 1 extends it to
  angular `bin_size`.
- `tests/test_argument_conventions.py` — the signature-rule tests. Phase 3
  adds one test per vocabulary rule.
- `tests/encoding/test_terminal_verbs_contract.py` — the result-verb
  contract. Phase 4 widens it to every public result class.
- `tests/data/public_api.txt` — the API snapshot. Phases 3a, 3b, 4 and 7
  change it. Regenerate with `NEUROSPATIAL_UPDATE_API_SNAPSHOT=1` and review
  the diff.
- `.claude/rules/public-api.md` and `CLAUDE.md` ("Canonical argument order",
  "Scientific conventions") — the written contract. Phases 3a, 3b and 7
  update them in the same PR as the code.
- Untouched: the numerics of every estimator. No phase changes a computed
  value on valid input, except where a default changes on purpose (phase 3b).
  Those changes are listed in the CHANGELOG.

## Scope and dependency policy

### Goals

- Remove every silent wrong result listed in the review (S1–S9).
- Make the quickstart and every simulator session usable with the
  package's own functions.
- Use one name and one meaning per argument, with each rule enforced by a test.
- Make every public result class meet `.claude/rules/public-api.md`.
- Make every error a user can hit name what failed, why, and the call to
  make instead.
- Let a user or LLM find the right entry point from `help()`/`dir()` and the
  docs.

### Non-Goals

- No deprecation shims, aliases or legacy keyword support. Rename in place
  (project policy: no users yet).
- No LLM redirect aliases (for example `compute_psth` → `peri_event_histogram`).
  The user rejected them as overkill.
- No broad rename of the `compute_` prefix. Phase 7 writes the rule down and
  renames only the outliers.
- No new analyses and no numerical-method changes.
- No animation (fastplotlib/napari) work beyond keeping keyword names
  consistent.
- No changes to the 930 weak private raise sites flagged by the AST heuristic,
  except the public-boundary ones listed in phase 6.

### Dependency policy

No new runtime dependencies. `difflib` (stdlib) is used for close-match hints.

## Metrics

- All nine S-cases from the review either warn (with numbers) or raise, as
  shown by the phase-1 tests.
- The golden-path workflow test emits **zero** warnings before and after
  every phase.
- The quickstart rate map has more than 80% finite bins, checked by the docs
  test.
- `tests/test_argument_conventions.py` has one test per rule in
  [shared-contracts.md#argument-vocabulary](shared-contracts.md#argument-vocabulary).
- The full default suite, the doctests and the executable docs pass at each
  PR boundary.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| New warnings fire on correct input (noise erodes trust) | Golden-path zero-warning test in phase 1, run in every later phase. Thresholds are derived in [designs.md](designs.md), not guessed. |
| Masked-time detection (S2) false-positives on real recordings with dropped frames | It only fires when the gaps are whole multiples of the median dt, cover more than 5% of duration, **and** stay under `max_gap`. It warns rather than raises, and is tested on a simulated dropped-frame recording. |
| Renames in 3a break docs, notebooks and examples | 3a includes a repo-wide grep task over `docs/`, `examples/`, the README and docstrings. The executable docs and doctests are the gate. |
| `criterion="shuffle"` default makes `is_*_cell` slow on first use | Documented in each docstring's first paragraph. `n_shuffles` default stated. The result-method screen is named as the fast path. |
| Phase 3 conflicts with concurrent work | 3a lands before 4–6. Phases 4, 5 and 6 touch mostly disjoint files and can run in parallel worktrees after 3b merges. |

## Rollout Strategy

Seven PRs (3 split into 3a/3b), merged in order 1 → 2 → 3a → 3b →
{4, 5, 6} → 7. Phases 4–6 may proceed in parallel. All changes take effect
immediately, with no flags and no deprecation period. Each PR adds entries
under "Unreleased" in `CHANGELOG.md`. Behavior changes (new raises, changed
defaults, renamed arguments) are listed as **Breaking**.

## Open Questions

1. Phase 3a: does a time-bin width outside the four listed functions still
   use `bin_size`? Current best answer: the phase-3a task greps for every
   `bin_size` whose docstring says seconds, and renames each one.
2. Phase 4: when called with no index, does a population `plot()` draw an
   overview (heatmap of units × bins) or raise the shared actionable error?
   Current best answer: draw an overview for PSTH and tuning results, where
   units × time or value has a natural heatmap; raise for spatial,
   directional and other 2-D maps, where there is no single-axes overview.
3. Phase 7: is `llms.txt` worth shipping? It is kept as a small, optional
   final task, generated from the snapshot with a sync test. Drop it if the
   phase grows.

## Estimated Effort

| Phase | Approx. diff |
| --- | --- |
| 1 | ~600 LOC src + ~500 LOC tests |
| 2 | ~250 src + docs |
| 3a | ~1,500 lines touched, mostly mechanical, plus docs/notebooks |
| 3b | ~500 src + tests |
| 4 | ~900 src + tests |
| 5 | ~500 src + tests |
| 6 | ~500 src + tests |
| 7 | ~400 docs/src + tests |
