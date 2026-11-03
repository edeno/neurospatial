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

4. **Resolved 2026-11-02: keep `min_speed`** on the analysis functions,
   default `None` (no hidden filtering). A visible `min_speed=` prompts the
   choice where the analysis is called; an `epochs=`-only path would not.
   Phase 3b must keep `min_speed=None` on the ten analysis functions. A
   follow-up after this plan: show the selection in result `repr`/`summary()`
   (analysed vs recorded seconds, and the gate that removed the rest), since
   today it is only in `analysis_settings`.

## Follow-ups recorded during execution

- **Per-interval path measures include the final step** (maintainer,
  2026-11-02). A trial `[start, stop)` excludes the step into the `stop`
  sample, so per-trial paths end one sample short of the goal edge and path
  efficiency can exceed 1 by at most one sample step. Apply "include the
  step to the interval's stop sample" to every per-interval path measure at
  once (path efficiency, goal-directed metrics, `path_progress`, and
  similar) through one shared helper, then tighten
  `test_path_efficiency_not_above_one` to `<= 1`.
- **Show the time selection in results** (min_speed decision): result
  `repr`/`summary()` report analysed vs recorded seconds and the gate that
  removed the rest.
- **Real-data survey findings not taken here:** G-030 (the NWB heading
  reader trusts a "degrees" label on radian data) and G-031 (headings
  squeezed into a small arc look like sharp head-direction cells; needs a
  calibrated spread threshold). See `datasets/GAPS.md`.
- **The other `*_significance` functions** still return dicts; give them
  population results like `PlaceCellSignificanceResult`.
- **`detect_laps` loop rule on `from_samples` arenas** (real-data survey,
  2026-11-02): unvisited patches inside an open arena count as holes, so
  running round an unvisited centre (DANDI 002030) gives a lap. Decide
  whether holes come only from declared walls/obstacles.
- **`TuningCurveResult.peak_location()`** returns a shape-(1,) array on 1-D
  tuning curves.
- **`simulation.mazes` was public but unpinned** until #267; check other
  public subpackages with an `__all__` are in `NAMESPACES`.
- **Rerun the "LLM guesses the API" probe** from a fresh session started
  outside this repo (the review's run had seen CLAUDE.md, so it is a lower
  bound).
- **`keyword_hint` has no hook for unexpected keywords**, so a result screen
  called with `criterion=` gives a plain `TypeError`.
- **Regressions found after the plan merged, both fixed:** #270 (the masking
  warning fired on irregular `times`) and #272 (spike times in milliseconds
  passed silently with `epochs` when tracking outlasts the spike recording).
  Under #272's rule, a seconds recording over 32 times longer than tracking
  that starts with it also warns; the maintainer accepted that.
- **Spike times in milliseconds can warn with the same text as correct
  units** (DANDI 001695): the "divide by 1000" hint in `_clock_ratio_hint`
  needs both ratios within 300–3,200, and 001695's span ratio is 5,800. The
  warning still fires, but nothing says milliseconds.
- **`test_a_hundred_thousand_bins_draw_in_seconds` fails on main**: the
  large-grid memory warning becomes an error. It is a wall-clock test the
  default marker filter excludes, so CI never runs it.

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
