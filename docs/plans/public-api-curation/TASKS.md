# Public API Curation Tasks

**Status:** Ready for staged execution

**Plan:** [PLAN.md](PLAN.md)

These tasks follow readiness dependencies rather than a release calendar. A
task is complete only when its acceptance evidence is linked or recorded.

## Agent Handoff

Use this prompt to resume the work in Claude Code, Codex, or another coding
agent:

> Read `CLAUDE.md`, `docs/plans/public-api-curation/PLAN.md`, this file, and
> `docs/plans/public-api-curation/artifacts/PROGRESS.md` if it exists. Preserve
> unrelated worktree changes. Execute the first unchecked, unblocked task in the
> document's execution order. Follow
> the authority boundaries and deterministic defaults below. Run the task-level
> checks, update the task checkbox and progress evidence, then stop at a
> reviewable boundary. Do not remove or hard-deprecate public API, change
> scientific semantics/defaults, add a required dependency, or create a new
> public facade without an accepted decision record.

## Agent Execution Contract

1. Read `CLAUDE.md` before editing and use `uv run` for every Python command.
2. Inspect `git status --short` before work. Existing changes belong to the
   operator; do not revert or overwrite unrelated files.
3. Execute the first unchecked, unblocked task in document order unless the
   operator names another task. Work on one task or one explicitly listed atomic
   batch at a time. Tasks may land independently; a phase is a readiness gate,
   not a release batch.
4. Mark the active task `[~]`. On completion, change it to `[x]` and append its
   evidence row to `artifacts/PROGRESS.md`. If blocked, leave it `[ ]`, record
   the blocker, and continue only with a task whose dependencies are satisfied.
5. Inspect before editing. Generated artifacts must identify their generator and
   must be reproducible with a checked-in command; do not hand-edit generated
   JSON or inventory tables.
6. Add or update focused tests with runtime changes. For public API changes,
   start with a small outside-in test through the canonical import path, then add
   integration or internal tests where the risk requires them. Run the narrowest
   relevant check during iteration and the phase gate before marking a phase
   complete.
7. Treat the current runtime behavior as the compatibility baseline unless a
   tested contract or accepted decision says otherwise. Do not convert an audit
   observation directly into a breaking change.
8. Do not add release numbers, release dates, or time-based removal promises.
9. Do not run user studies. Discovery evaluation is limited to the objective
   checks and maintainer walkthroughs defined here.

### Authority boundaries

| Change | Agent action |
| --- | --- |
| Add inventories, reports, documentation, tests, curated policy metadata, or internal audit/benchmark tooling | Implement autonomously when the task authorizes it. |
| Correct an artifact to match already-tested runtime behavior | Implement and record the evidence. |
| Correct a stale deprecation version/date in prose or a warning while retaining the compatibility path, warning category, and replacement guidance | Implement with focused tests; this is a truthfulness fix, not removal authority. |
| Choose among classifications when the rules below determine a conservative answer | Apply the rule and record the rationale. |
| Remove a public path, add a hard-deprecation warning, or intentionally break a stable contract | Create a pending decision record and stop that change until it is accepted. |
| Change a scientific default, coordinate meaning, unit convention, missing-data behavior, or numerical result | Create a pending scientific decision record and stop that change until it is accepted. |
| Add a required dependency, materially expand an optional integration, or introduce a new public facade/container | Send it to the gap report and a separate design decision; do not implement it here. |

An unresolved gated decision blocks only the dependent mutation. Inventory,
documentation of current behavior, additive contract tests, and unrelated tasks
continue.

### Deterministic defaults

Use these rules instead of asking for routine classification choices:

| Ambiguity | Default |
| --- | --- |
| Public reachability is unclear | Anything in an `__all__`, documented public path, or supported compatibility path remains supported until reviewed. Record separately whether it is recommended/exported, documented, compatibility-only, and canonical; do not add compatibility-only names to `__all__` merely to preserve support. |
| Stable versus advanced is unclear | Classify as advanced; promotion to stable requires all stable evidence. |
| Advanced versus experimental is unclear | Use experimental only when the interface itself is unsettled; specialized but mature functionality is advanced. |
| An alternate import path may be removable | Keep it supported, warning-free, and classify it as permanent-alias or soft-deprecated pending a gated decision. |
| Adapter preservation is unproven | Classify it as extraction, never lossless. |
| Core or optional dependency range is unclear | Record the declared lower bound and the locked/tested version; if no lower bound exists, record only the locked/tested version. Do not invent an upper bound or support claim. |
| Array/backend behavior is untested | Promise NumPy behavior only and record other protocols as unverified. |
| Coordinate semantics are unclear | Record representation, reference frame, units, and array indexing separately; do not collapse them into a generic coordinate claim. |
| An identifier may be a positional index | Preserve the existing value and label the distinction as unresolved; do not renumber or reinterpret it during inventory. |
| Multi-segment behavior is unclear | Preserve segments and record concatenation as unsupported/unverified; never infer silent concatenation. |
| Partial loading or sparsity is unproven | Record full materialization/dense behavior or “unknown”; do not claim a partial read or sparse-preserving conversion. |
| Scale behavior is unmeasured | Record “unknown”; do not claim bounded memory or acceptable performance. |
| Namespace variants are close | Keep the current runtime namespace and improve documentation/discovery; do not add a second facade. |
| Existing docs and runtime disagree | Treat tested runtime as current behavior, open a decision record if scientific intent is unclear, and do not silently rewrite semantics. |
| A custom class, stateful workflow, or broad protocol seems unnecessary | Record the simpler function/standard-type alternative and current invariant or state requirement; do not redesign the public API without an accepted decision. |

### Required work products

Create paths when their producing task begins. Every Markdown artifact must state
its generating task, last update date, and whether it is generated or manually
reviewed.

All paths beginning with `artifacts/` are relative to
`docs/plans/public-api-curation/`; all other paths are relative to the repository
root.

| Path | Contents | Producing tasks |
| --- | --- | --- |
| `artifacts/PROGRESS.md` | Task status, changed files, commands, results, decisions, and blockers | A.0; updated by every task |
| `artifacts/DECISIONS.md` | Numbered gated decisions with status, recommendation, alternatives, evidence, and affected tasks | A.0, 0.12; updated as needed |
| `artifacts/TOP_LEVEL_CLASSIFICATION.md` | Initial classification and rationale for every current top-level export, explicitly including all lazy exports | A.5, 0.2 |
| `artifacts/RNG_AUDIT.md` | Public `rng`/`seed` vocabulary, accepted types, determinism behavior, private adapter terminology, and migration candidates | A.6, 1.4, 3.4 |
| `docs/api/public_api_metadata.schema.json` | Machine-validated schema for manually curated, non-derivable API policy | 0.1 |
| `docs/api/public_api_metadata.json` | Canonical paths, tiers, aliases, lifecycle, owning domains, profiles, extras, and compatibility conditions | 0.1-0.2, expanded to the complete supported surface by 1.2 and finalized by 1.12 |
| `docs/api/conventions.md` | Design grammar, data-semantics/lazy-source and result/contract profiles, lifecycle, and exceptions | 0.3-0.7, 0.10-0.11 |
| `artifacts/DEPRECATION_LEDGER.md` | Active aliases/warnings, replacements, conditions, evidence, and disposition | A.1, 0.8-0.9, 3.1-3.3 |
| `scripts/audit_public_api.py` | Deterministic inventory generator and checker | 1.1-1.6 |
| `tests/test_audit_public_api.py` | CLI, determinism, drift, and actionable-error tests for the inventory generator | 1.1 |
| `artifacts/api_inventory.json` | Machine-readable generated inventory | 1.1-1.12 |
| `artifacts/API_INVENTORY.md` | Human-readable findings, data-semantics/lazy-source profiles, and unresolved classifications | 1.2-1.12 |
| `artifacts/INTEROPERABILITY_MATRIX.md` | Version, import, representation/frame/unit, ID/index/segment, partial-read/cache, preservation/loss, schema, and resource-lifetime matrix | 1.8-1.9, 2.8, 4.10 |
| `scripts/benchmark_public_api.py` | Reproducible population-scale scenarios and report writer | 4.11-4.12 |
| `artifacts/SCALE_BASELINE.md` | Parameters, environment, storage/chunk shape, peak memory, time, materialization/densification, allocations, and limitations | 4.11-4.12 |
| `artifacts/NAMESPACE_EVALUATION.md` | Provisional namespace default plus any compared variants, journey walkthroughs, objective checks, and decision | 1.10, 2.10 |
| `docs/api/researcher-api.md` | Task-oriented canonical API map | A.3, refined by 2.1-2.6 |
| `tests/typing/top_level_lazy_exports.py` | Static-type fixture proving headline lazy imports are not `Any` | A.2 |
| `docs/migration/public-api.md` | Version-agnostic alias and migration table | 3.1-3.8 |
| `tests/data/public_api_contract.json` | Generated, human-reviewable stable snapshot | 4.1 |
| `tests/public_api/test_researcher_workflows.py` | Readable outside-in contract tests using canonical public imports | 4.6 |
| `tests/test_api_semantics.py` | Behavioral and metamorphic contract tests for declared semantic dimensions | 4.7-4.9 |
| `tests/test_interoperability_contract.py` | Core and optional integration preservation/loss contract tests | 4.10 |
| `artifacts/GAP_REPORT.md` | Out-of-scope API or algorithm proposals | Phase 1 onward; finalized in Phase 6 |

`artifacts/PROGRESS.md` uses one row per task:

```text
| Task | Status | Changed files | Validation and result | Decisions/blockers |
```

`artifacts/DECISIONS.md` uses one section per decision:

```text
## D-001: Short decision title
Status: pending | accepted | rejected | superseded
Gate: compatibility | scientific semantics | dependency | new public design
Recommendation: ...
Alternatives: ...
Evidence: ...
Affected tasks/files: ...
Maintainer decision: ...
```

Only a maintainer changes a gated decision from `pending` to `accepted` or
`rejected`. An agent may mark a decision `superseded` when later repository
evidence proves that no gated change is needed, and must record that evidence.

### Dependency and phase order

```text
A.0 bootstrap
  -> Tranche A immediate researcher-facing corrections
      -> Phase 0 policy/metadata-schema draft
      -> Phase 1 reproducible inventory
          -> Phase 2 additive docs/discovery
              -> Phase 3 approved canonicalization/migrations
                  -> Phase 4 enforcement
                      -> Phase 5 separately approved cleanup
                          -> Phase 6 closeout/gap disposition
```

- Tranche A tasks land independently and must not wait for Phase 0 governance.
- Phase 1 requires 0.1-0.4, 0.7, and 0.10-0.12. Pending namespace or
  deprecation decisions do not block inventory.
- Phase 2 requires a reproducible Phase 1 inventory and draft classifications.
- Namespace comparison is non-blocking: the current sparse runtime namespace is
  the provisional default, and an unfinished comparison cannot block the task
  page, primary navigation cleanup, or other documentation work.
- Each Phase 3 runtime change requires its own accepted decision record. Phase
  3 metadata, migration docs, and non-breaking tests may proceed without one.
- Phase 4 can guard unchanged behavior while Phase 3 decisions are pending.
- Phase 5 is never implied by completion of earlier phases; each removal needs
  an accepted decision and satisfied compatibility conditions.
- Phase 6 consolidates findings only; material additions move to separate plans.

### Definition of done for one task

A task is complete only when all applicable statements are true:

- Its named files exist and contain no placeholders such as “TBD” unless the
  item is explicitly recorded as unknown or pending in `DECISIONS.md`.
- Generated output is reproducible and `--check` detects drift where specified.
- Focused tests cover success and failure behavior for code changes.
- Relevant formatting, lint, type, test, or documentation commands pass.
- `git diff --check` passes and unrelated worktree files are untouched.
- The task checkbox and `artifacts/PROGRESS.md` agree.

### Validation protocol

Use the applicable task-level checks while iterating:

```bash
git diff --check
uv run ruff check <changed-python-files>
uv run ruff format --check <changed-python-files>
uv run pytest -n 0 <focused-test-files>
uv run python scripts/test_doc_snippets.py
uv run mkdocs build --strict
```

Do not pass Markdown or JSON files to Ruff. When optional-integration testing is
required, prepare only the relevant extras:

```bash
uv sync --extra dev --extra docs --extra pynapple --extra nwb --extra xarray
```

Record unavailable platforms or external system dependencies as an untested
matrix row; never convert a skipped check into a support claim.

Run these phase gates before checking a phase's acceptance boxes:

| Phase | Required commands |
| --- | --- |
| Tranche A | `uv run pytest -n 0 tests/behavior/test_detect_region_crossings_argorder.py tests/test_lazy_imports.py tests/test_sparse_init_exports.py tests/test_no_internal_doc_refs.py tests/test_doc_snippets_helper.py`; `uv run mypy tests/typing/top_level_lazy_exports.py`; `uv run python scripts/test_doc_snippets.py`; `uv run mkdocs build --strict`; `git diff --check` |
| 0 | `uv run pytest -n 0 tests/test_public_api_contract.py`; `git diff --check` |
| 1 | `uv run python scripts/audit_public_api.py --check`; `uv run pytest -n 0 tests/test_audit_public_api.py tests/test_public_api_contract.py`; `git diff --check` |
| 2 | `uv run pytest -n 0 tests/test_no_internal_doc_refs.py tests/test_doc_snippets_helper.py`; `uv run python scripts/test_doc_snippets.py`; `uv run mkdocs build --strict`; `git diff --check` |
| 3 | `uv run pytest -n 0 tests/test_public_api_contract.py` plus the focused domain tests recorded for each approved migration; `git diff --check` |
| 4A | `uv run python scripts/audit_public_api.py --check`; `uv run pytest -n 0 tests/test_public_api_contract.py tests/test_lazy_imports.py tests/test_package_imports.py tests/test_no_internal_doc_refs.py`; `uv run mypy src/neurospatial`; `git diff --check` |
| 4B | Run all commands under **Expected Validation Commands**, followed by `uv run pytest -n 0`; `git diff --check` |
| 5 | Re-run the Phase 4A gate and every applicable Phase 4B check after regenerating artifacts. |
| 6 | Verify every gap-report entry has a disposition, then run `git diff --check`. |

If a listed test file belongs to the current phase and does not exist yet, the
task that first needs it creates it. A phase gate is not waived because its test
file is absent.

## Tranche A: Immediate researcher-facing corrections

**Purpose:** deliver current, visible UX fixes before building the full
governance apparatus. These tasks are individually mergeable and do not grant
authority for API removal, scientific changes, or a new facade.

- [ ] **A.0 Bootstrap execution records.** Create `artifacts/PROGRESS.md` and
  `artifacts/DECISIONS.md` from the schemas above. Record the initial worktree
  state and mark A.0 complete. Do not modify runtime code in this task.
- [ ] **A.1 Correct the stale 0.7 deprecation promise.** In
  `detect_region_crossings`, retain the old call form and its
  `DeprecationWarning`, but remove the false version/time removal promise from
  the docstring, warning, TODO, and focused tests. Keep the replacement call and
  `stacklevel` guidance intact. Seed `artifacts/DEPRECATION_LEDGER.md` with this
  compatibility path; do not remove it in this task.
- [ ] **A.2 Restore static typing for lazy top-level exports.** Make
  `SpikeTrains`, `Session`, `load_session`, `BayesianDecoder`, and `restrict`
  resolve to their concrete types under mypy while preserving PEP 562 runtime
  laziness and minimal-import behavior. Add
  `tests/typing/top_level_lazy_exports.py`; validate it with mypy and retain the
  runtime lazy-import tests.
- [ ] **A.3 Draft `docs/api/researcher-api.md`.** Map the nine existing
  researcher journeys to current canonical entry points. Lead systems-
  neuroscience workflows with population paths, label classifications as
  provisional until the curated metadata lands, and use executable examples
  where the existing snippet harness supports them.
- [ ] **A.4 Remove private modules from primary API navigation.** Update
  `docs/gen_ref_pages.py` so underscore-prefixed/private modules are absent from
  the primary navigation. Preserve generated pages that are still link targets
  until link checks prove they can be deleted. Run strict docs and internal-link
  checks.
- [ ] **A.5 Classify the current top-level surface explicitly.** Create
  `artifacts/TOP_LEVEL_CLASSIFICATION.md` for every current top-level export.
  Call out `Session`, `load_session`, `SpikeTrains`, `BayesianDecoder`,
  `restrict`, and the lazy submodules individually. Use owning domain plus the
  project maintainer by default; record uncertainty conservatively without
  changing runtime exports.
- [ ] **A.6 Audit public RNG vocabulary without renaming it.** Create
  `artifacts/RNG_AUDIT.md` separating public `rng` APIs from public `seed` APIs
  and private adapter parameters such as sklearn-facing `random_state`. Record
  accepted types, determinism/substream behavior, docs/tests, and compatible
  migration options. Do not rename a public keyword in this task.

### Tranche A acceptance

- [ ] Active docs and warnings no longer promise removal in 0.7, while the old
  `detect_region_crossings` call remains tested and warning-compatible.
- [ ] Mypy reveals concrete types—not `Any`—for all five lazy top-level exports,
  and runtime imports remain lazy.
- [ ] The Researcher API page and cleaned primary navigation build strictly.
- [ ] The top-level classification explicitly covers `Session` and every other
  lazy export.
- [ ] The RNG audit distinguishes public vocabulary from private adapter names;
  no public RNG keyword changed in this tranche.
- [ ] No public path, scientific behavior, or compatibility shim was removed.

## Phase 0: Governance and deprecation debt

**Dependencies:** Tranche A. Its classifications and audits seed the curated
metadata, grammar, and ledgers rather than being repeated.

- [ ] **0.1 Define the curated metadata schema.** Add
  `docs/api/public_api_metadata.schema.json` and
  `docs/api/public_api_metadata.json`. Include only reviewed policy that cannot
  be safely generated: tier, owning domain, optional owner override, canonical
  path, intentional aliases, internal documentation exclusions,
  optional-extra requirements,
  grammar/result/contract profiles, lifecycle state, interoperability guarantees,
  and deprecation conditions. Do not duplicate signatures, object kind, or other
  discoverable structure. The project maintainer is the default owner; do not
  repeat that value on every entry. Start `tests/test_public_api_contract.py`
  with deterministic schema/entry validation using `jsonschema`; declare it in
  the `dev` extra if it is not already a direct development dependency. Loading
  these JSON files must not add an installed-package dependency or import
  neurospatial domain or optional packages.
- [ ] **0.2 Record the current top-level contract.** Import the conservative
  classifications from `artifacts/TOP_LEVEL_CLASSIFICATION.md`, explicitly
  including `Session`, `load_session`, and the other lazy exports. Add all
  current top-level exports without removing runtime paths.
- [ ] **0.3 Write the API design grammar.** Add `docs/api/conventions.md` with
  canonical vocabulary, function-family templates, signature rules, raw and
  labeled dimensions, time/epoch semantics, copy/mutation/ownership behavior,
  dtype/device coercion, accepted array protocols, RNG conventions, and
  warning/error patterns. Include explicit rules for I/O/computation separation,
  friendly normalization versus strict kernels, required scientific choices,
  stable return kinds, interacting parameters, documented `**kwargs` forwarding,
  structural `Protocol` claims, custom-class justification, and explicit fitted
  state. Define data-semantics profiles that separate coordinate representation,
  reference frame, physical units, and array indexing; distinguish stable IDs
  from positional indices and segments from concatenated time; and define
  dtype/value-range and lazy-source vocabulary.
- [ ] **0.4 Define result profiles.** Specify labeled tensor/map,
  tabular/event, fitted-model, scalar/report, and streaming/chunked profiles;
  require only meaningful conversion and terminal methods. The fitted-model
  profile distinguishes specification, `fit()` mutation/return behavior, learned
  state inspection, pre-fit errors, prediction/decode/transform behavior, and
  resolved configuration/provenance.
- [ ] **0.5 Draft tier semantics and promotion evidence.** Document the
  change process and readiness criteria for stable, advanced, experimental,
  and internal APIs, including documentation, interoperability, scale, owning
  domain, and evaluation requirements. Use the project maintainer as the
  default accountable person and add an override only when responsibility truly
  differs.
- [ ] **0.6 Draft the top-level policy and namespace comparison.** Classify
  which current conveniences are permanent, compatibility-only, or candidates
  for canonical subpackage paths. Treat the current sparse runtime namespace as
  the provisional default. Define a non-blocking comparison with a small
  task-facade prototype without first adding runtime aliases.
- [ ] **0.7 Define lifecycle states.** Separate recommended, soft-deprecated,
  hard-deprecated, and permanent-alias states from support tiers. State that
  soft deprecation carries no warning or removal promise.
- [ ] **0.8 Audit active deprecations.** Produce a ledger of every warning,
  advertised removal condition, replacement, docs reference, and current use,
  extending the ledger seeded by A.1.
- [ ] **0.9 Reconcile overdue notices in the ledger.** Recommend retain, revise,
  or remove for every remaining entry. Non-breaking corrections to false
  version/date text may proceed under the A.1 rule; warning category changes,
  behavior changes, and removals wait for an accepted decision.
- [ ] **0.10 Define contract profiles.** Specify how array, scientific,
  missing-data, result, mutation/ownership, execution/coercion, diagnostic,
  reproducibility, interoperability, optional-dependency, and scale dimensions
  are represented. Include spatial-array, population, segmented-time, and
  lazy-source profiles as contracts, not proposed base classes.
- [ ] **0.11 Define interoperability and loss policy.** Specify lossless,
  extraction, and lossy categories; separate core-dependency, optional-integration,
  and array-protocol rows; distinguish declared support from tested versions;
  define the default tested cells as the declared minimum plus locked/current
  version (or locked/current only when no minimum is declared); and specify
  reversible-schema claims and lazy resource-ownership requirements.
- [ ] **0.12 Implement the decision workflow.** Add compatibility, scientific,
  dependency, and new-public-design gates to `artifacts/DECISIONS.md`. State the
  evidence required for each and identify the project maintainer as approver.

### Phase 0 acceptance

- [ ] Curated metadata and its schema load and validate without importing any
  neurospatial domain or optional package.
- [ ] Every current top-level export is represented.
- [ ] API grammar and result profiles are internally consistent, and any
  scientific exceptions or gated choices are recorded explicitly.
- [ ] Data-semantics profiles independently represent coordinate
  representation/frame, units, axes, dtype/range, IDs/indices, segments, and
  lazy-source behavior without requiring a new runtime abstraction.
- [ ] Curated metadata can link grammar/result profiles, lifecycle state, and
  interoperability guarantees without importing optional packages.
- [ ] Deprecation ledger has no untriaged entry.
- [ ] No public symbol was removed solely to close Phase 0.

## Phase 1: Inventory and contract proposal

**Dependencies:** Phase 0 schema, grammar, lifecycle, contract/loss policy, and
decision workflow. Pending gated decisions use the conservative defaults above.

- [ ] **1.1 Implement `scripts/audit_public_api.py`.** Inventory top-level and
  subpackage `__all__`, lazy mappings, static declarations, runtime exports,
  documented public paths, supported compatibility paths, and internal-module
  documentation-exclusion candidates reproducibly, then validate them against
  curated policy metadata. Keep recommended/exported, documented,
  compatibility-only, canonical, and excluded-internal status as separate fields.
  `--write --allow-incomplete-policy` must generate
  `artifacts/api_inventory.json` and `artifacts/API_INVENTORY.md`; `--check`
  must be the strict normal mode and exit nonzero for stale output, missing policy
  entries, or contradictory declarations. The incomplete-policy mode must emit a
  deterministic `missing_policy_entries` list in the JSON inventory plus a
  matching human-readable report section. It may permit missing entries only; it
  may not silently classify them or suppress contradictory-entry failures. Add
  `tests/test_audit_public_api.py` for deterministic ordering, incomplete-policy
  bootstrap, strict missing-policy failure, clean strict `--check`, and
  stale-output failure behavior. Task 1.1 is complete when the incomplete-policy
  bootstrap and the expected strict failure are verified; a clean strict check
  first becomes required in task 1.2.
- [ ] **1.2 Capture discovered structure and complete curated policy.** Generate
  module, object kind, signature, runtime aliases, export mechanism, documentation
  reachability, and compatibility reachability from the package and repository.
  Use the deterministic missing-policy queue to add every supported symbol and
  path plus every reviewed internal documentation exclusion to
  `docs/api/public_api_metadata.json`, applying the conservative defaults and
  recording genuine breaking or scientific ambiguities in
  `artifacts/DECISIONS.md`. Merge owning domain, optional owner override,
  tier/lifecycle proposal, canonical path, grammar/result profile, intentional
  aliases, and optional dependency from the completed metadata. Do not copy
  generated facts into that file. Regenerate without the incomplete-policy flag
  and require strict `--check` to pass before completing the task.
- [ ] **1.3 Map evidence.** Record doc, example, test, changelog, and migration
  references. Mark automated mappings as best-effort when static inference is
  ambiguous.
- [ ] **1.4 Audit API-family grammar.** Compare recurring parameter names,
  positional/keyword-only controls, singular/plural and dimensional families,
  compute/fit/decode/detect/convert verbs, required scientific choices, hidden
  or forwarded `**kwargs`, interacting mode parameters, boolean return-type
  flags, stable return kinds, `rng` / `seed`, and deliberate scientific
  exceptions. Extend `artifacts/RNG_AUDIT.md`; do not count private backend
  parameters such as sklearn-facing `random_state` as public API inconsistencies.
- [ ] **1.5 Audit architectural fit and observable contracts.** For each stable
  candidate, classify its boundary role as I/O/adapter, friendly normalization,
  strict scientific computation, result/presentation, or deliberate combination.
  Record optional-dependency leakage, file access inside computation, concrete
  type checks that reject otherwise valid interfaces, claimed/tested Protocols,
  custom classes that could use a standard type or dataclass, and methods invalid
  before hidden state transitions. Do not redesign from this audit; route material
  changes through `artifacts/DECISIONS.md` or the gap report. Also add applicable
  shape, axis, coordinate representation/reference frame, physical units/time
  support, missing-data, result, copy/view/mutation, resource ownership,
  eager/lazy, dtype/value-range/device coercion, accepted array protocol,
  warning/error, return-kind, RNG, serialization, and scale behavior. For lazy
  or file-backed candidates, record pre-load shape/dtype/identity, supported
  partial slicing, compute trigger, cache policy, dtype/scaling, state inspection,
  file ownership/closure, and operations unavailable before materialization.
  For graph-returning APIs, record node identity and whether the result is a
  view, copy, or shared mutable object.
  Flag catch-and-print paths and require diagnostics to identify the bad value or
  input, violated condition, and corrective action.
- [ ] **1.6 Audit result profiles and schemas.** Identify stable result fields,
  summary keys, methods, DataFrame identifiers, xarray dimensions/coordinates,
  essential data currently stored only as attrs, NWB round-trip names, fitted
  configuration versus learned state, `fit()` mutation/return behavior, pre-fit
  errors, and resolved-parameter provenance.
- [ ] **1.7 Audit identity, ordering, and segments.** Trace stable `unit_id`
  values versus positional `unit_index`, node/bin identity versus array position,
  and `segment_id` versus `segment_index` through batch computation, selection,
  filtering, reordering, iteration, summary conversion, persistence, and
  interoperability. Record output ordering and whether multi-segment inputs are
  preserved, explicitly aligned/concatenated, rejected, or unverified; never
  silently infer concatenation.
- [ ] **1.8 Audit Pynapple preservation.** For `TsGroup`, `Tsd`, `TsdFrame`,
  and `IntervalSet`, trace unit IDs, metadata, columns, time support, units, and
  interval semantics through direct inputs and adapters. Also trace ID/index and
  segment distinctions plus coordinate representation/reference frame where
  applicable. Classify each path as lossless, extraction, or lossy.
- [ ] **1.9 Build the interoperability matrix.** Record pandas as a required-core
  schema integration; Pynapple, pynwb/NWB, and xarray as optional integrations;
  and accepted array protocols as execution/coercion contracts. Separate declared
  support from tested cells, using the declared minimum and locked/current version
  by default, or locked/current only when no minimum is declared. For applicable
  rows record import, direct-input, conversion, round-trip,
  representation/frame/unit preservation, ID/index and segment preservation,
  pre-load structure, partial reads, caching, lazy materialization, dtype/scaling,
  and file-lifecycle coverage.
- [ ] **1.10 Record the namespace default and optional comparison.** Document
  the current sparse runtime namespace as the provisional default in
  `artifacts/NAMESPACE_EVALUATION.md`. If a small task-facade
  documentation/autocomplete prototype is cheap to produce, measure `dir()`,
  static-type discoverability, searchability, and import cost; otherwise record
  the comparison as deferred. Deferral does not block Phase 1 or 2.
- [ ] **1.11 Map researcher journeys.** For every journey in the plan, identify
  canonical input object, entry point, result profile, identity/time-support
  behavior, coordinate/frame/unit and segment semantics, scale/materialization
  behavior, and next step.
- [ ] **1.12 Finalize agent-safe classifications.** Apply the deterministic
  defaults, resolve classifications supported by repository evidence, persist the
  resulting policy fields in `docs/api/public_api_metadata.json`, and put every
  breaking or scientific ambiguity in `artifacts/DECISIONS.md`. Pending decisions
  retain their conservative supported classification and do not block the
  inventory. Regenerate the inventory and finish with a strict `--check` so no
  supported path lacks policy metadata.

### Phase 1 acceptance

- [ ] Re-running the audit reproduces baseline counts.
- [ ] Every proposed stable symbol has an owning domain, grammar/result profile,
  and contract profile, or a reviewed reason a profile is inapplicable. The
  project maintainer is the default owner unless an override is recorded.
- [ ] Every grammar deviation and scientific exception is recorded rather than
  silently normalized.
- [ ] Every proposed stable custom class, stateful method family, structural
  protocol, and mixed I/O/computation boundary has a recorded justification or a
  gated follow-up; the inventory itself does not trigger a redesign.
- [ ] Every applicable stable array/container records representation, frame,
  units, axes, dtype/range, IDs versus indices, ordering, segment behavior, and
  lazy-source semantics or a reviewed inapplicable rationale.
- [ ] Every stable graph-returning API declares node identity plus view/copy and
  shared-mutation behavior.
- [ ] Every core or optional integration has separate declared-support and
  tested-version rows, every adapter has a loss classification, and array
  protocols are recorded as execution/coercion contracts rather than dependencies.
- [ ] Known population allocations and scale risks are recorded where evidence
  already exists; unknowns are explicitly deferred to Phase 4 rather than
  guessed.
- [ ] Inventory ambiguities and unresolved classification decisions are
  recorded, not silently guessed away.
- [ ] No public symbol was removed solely to close Phase 1.

## Phase 2: Documentation and discoverability

**Dependencies:** reproducible Phase 1 inventory and draft classifications.
This phase is additive and does not change runtime canonical paths.

- [ ] **2.1 Refine `docs/api/researcher-api.md`.** Replace A.3's provisional
  classifications with metadata-backed tiers and canonical paths while
  preserving its working journey examples.
- [ ] **2.2 Generate reference navigation from exports and curated metadata.**
  Update `docs/gen_ref_pages.py` so primary navigation combines discovered
  public exports with tiers and canonical paths from
  `docs/api/public_api_metadata.json`, rather than following filesystem discovery
  alone. Preserve A.4's exclusion of private modules; advanced entries remain
  searchable and internals stay out of primary navigation.
- [ ] **2.3 Separate support tiers visually.** Stable and advanced APIs remain
  searchable; experimental APIs carry a clear badge; internals are absent.
- [ ] **2.4 Canonicalize teaching imports.** Examples and guides teach one path
  while identifying compatibility aliases only in migration material.
- [ ] **2.5 Lead with population paths.** Show singular and batch workflows
  together, using population entry points first where appropriate.
- [ ] **2.6 Document scale at the call site.** State major output dimensions,
  coordinate representation/frame and units, ID/index and segment semantics,
  dtype/materialization behavior, memory risks, practical limits, and existing
  summary/blocked alternatives. If no practical alternative exists, link its
  dispositioned gap-report entry. If a conversion densifies naturally sparse or
  unit-specific data, state the resulting allocation.
- [ ] **2.7 Publish API conventions.** Document the design grammar, result
  profiles, lifecycle labels, canonical identifiers/dimensions, temporal rules,
  coordinate representation/frame, units, ID/index and segment rules,
  dtype/value ranges, lazy-source profile, RNG convention, mutability/ownership,
  coercion, fitted state, and reviewed exceptions.
- [ ] **2.8 Publish interoperability contracts.** Update
  `docs/user-guide/interoperability.md` from
  `artifacts/INTEROPERABILITY_MATRIX.md`: document supported/tested dependency
  versions, Pynapple/NWB preservation and loss, reversible versus
  presentation-only schemas, coordinate/frame/unit and segment preservation,
  partial reads, caching, dtype/scaling, lazy compute triggers, state-dependent
  operations, and file-handle ownership.
- [ ] **2.9 Keep lazy namespaces discoverable.** Extend A.2's typing fix across
  lazy submodules and future lazy exports. Verify `dir()`, documentation search,
  IDE completion, and static typing; add or generate type stubs only where the
  lazy-loading mechanism requires them.
- [ ] **2.10 Run task and namespace discovery evaluation.** Compare the tested
  namespace-first and small task-facade prototypes through maintainer
  walkthroughs of the documented journeys and objective runtime, static-tool,
  documentation-search, and import-cost checks. Record inputs, commands, raw
  results, and the recommendation in `artifacts/NAMESPACE_EVALUATION.md`. If the
  optional prototype was deferred, record the current sparse namespace as the
  maintained default and close this task without blocking documentation. Do not
  retain both as permanent public paths by default.

### Phase 2 acceptance

- [ ] Every stable symbol has an anchored reference entry.
- [ ] Every researcher journey has executable documentation.
- [ ] No internal module appears in primary API navigation.
- [ ] Documentation checks validate links, anchors, and executable snippets.
- [ ] Every claimed round trip has an executable preservation test or is
  explicitly labeled extraction/non-reversible.
- [ ] Every applicable stable task page exposes representation/frame/units,
  axes, IDs/indices, segments, and materialization behavior without requiring the
  reader to infer them from implementation details.
- [ ] Lazy public namespaces resolve in runtime discovery and static tooling.
- [ ] Namespace comparison results are recorded if performed; deferral retains
  the current sparse namespace and does not block Phase 2 acceptance.

## Phase 3: Canonical paths and migrations

**Dependencies:** Phases 1-2. Each runtime-facing mutation in this phase needs
an accepted decision record; documentation and metadata can proceed around
pending decisions.

- [ ] **3.1 Resolve duplicated exports.** Choose a canonical path and classify
  every alternate as permanent or migratory.
- [ ] **3.2 Add migration metadata.** Record replacement path and eligibility
  conditions plus recommended/soft-deprecated/hard-deprecated/permanent-alias
  lifecycle state for every alternate.
- [ ] **3.3 Add policy-compliant warnings.** Test category, message, stack level,
  replacement behavior, and feedback route where hard-deprecation warnings are
  required. Do not warn merely because an API is soft-deprecated.
- [ ] **3.4 Normalize high-impact grammar inconsistencies.** Limit changes to
  issues that obstruct task discovery or generic population workflows:
  recurring names, keyword-only controls, `rng` conventions, array/dimension
  identifiers, coordinate representation/frame and units, ID/index and segment
  semantics, dtype/range and lazy-source behavior, temporal semantics,
  copy/mutation behavior, and explicit conversions rather than return-type flags.
- [ ] **3.5 Align result profiles and schemas.** Normalize cross-domain unit
  identifiers and labeled dimensions, keep stable IDs separate from positional
  indices, preserve segment labels, keep distinct scientific representations and
  frames distinct, make fitted state explicit, and remove only meaningless
  inherited result methods through the compatibility process.
- [ ] **3.6 Resolve approved interoperability losses.** Preserve metadata/time
  support/units where the existing surface can do so compatibly; otherwise
  relabel the adapter as extraction and send any richer API to follow-on design.
- [ ] **3.7 Verify equivalence contracts.** Preserve documented singular/batch,
  full/chunked, arrays/Pynapple, and result/conversion numerical agreement,
  identity, units, time support, and missing-data meaning.
- [ ] **3.8 Publish `docs/migration/public-api.md`.** Keep canonical and
  compatibility paths, lifecycle state, grammar/schema changes, and
  preservation effects explicit without promising a release or removal date.

### Phase 3 acceptance

- [ ] Active documentation and examples use canonical paths only.
- [ ] Every alternate public path is classified and tested appropriately.
- [ ] Soft-deprecated paths remain warning-free, documented, and tested unless
  separately approved for hard deprecation.
- [ ] Grammar and schema migrations preserve scientific meaning or document the
  reviewed change explicitly.
- [ ] No stable contract changed without the approved compatibility process.

## Phase 4: Contract and scale enforcement

**Dependencies:** Phase 1 inventory. Guards for unchanged behavior may land
before all Phase 3 decisions; guards for changed behavior require the relevant
accepted decision.

### Stage 4A: Low-cost structural guards

These checks land first. They protect the curated surface without constructing
the full behavioral/interop/benchmark matrix.

- [ ] **4.1 Generate `tests/data/public_api_contract.json`.** Include signatures
  and explicitly selected schema/behavior metadata. The audit script owns this
  file and `--check` reports semantic field changes, not only a hash mismatch.
- [ ] **4.2 Expand `tests/test_public_api_contract.py`.** Compare curated metadata,
  `__all__`, lazy mappings, static declarations, runtime resolution, canonical
  paths, and the generated contract snapshot.
- [ ] **4.3 Add documentation consistency tests.** Fail on missing stable
  references, undocumented stable exports, or internal primary-nav entries.
- [ ] **4.4 Add minimal-install import tests.** Ensure optional stacks are not
  loaded by core imports and missing extras fail actionably.
- [ ] **4.5 Add lazy-discovery checks.** Exercise normal and eager-import modes,
  `dir()`, declared lazy names, type stubs, and representative static-type/IDE
  resolution so postponed import errors are caught in CI.

#### Stage 4A acceptance

- [ ] Metadata/export, documentation, minimal-import, lazy-runtime, and
  lazy-static-typing drift fail with actionable messages.
- [ ] These guards run without installing unrelated optional stacks.
- [ ] Stage 4A can land and protect the surface while Stage 4B remains in
  progress.

### Stage 4B: Declared semantic, interoperability, and scale guards

Start with the supported researcher behavior, then add a guard only when curated
metadata declares that contract dimension for the symbol or workflow. Do not
generate the Cartesian product of every compatibility dimension and every stable
symbol. A dimension may be recorded as inapplicable or unverified with a
rationale rather than receiving a meaningless test.

- [ ] **4.6 Add public-interface workflow tests.** Create
  `tests/public_api/test_researcher_workflows.py` with at least one small,
  representative outside-in test for every stable researcher journey. Import
  only canonical public paths, exercise real collaborators rather than private
  attributes, keep setup readable, and mock only external or nondeterministic
  boundaries. Each test should demonstrate supported inputs, primary output,
  coordinate/frame/unit, identity/index, segment/time-support, and materialization
  behavior plus the next public operation where applicable.
  Keep optional Pynapple/NWB version and preservation cases in task 4.10; this
  suite covers their dependency-free journey alternatives and may select marked
  optional cases only in the matching extra environment.
- [ ] **4.7 Add `tests/test_api_semantics.py`.** Guard stable
  coordinate representation/reference frame and unit conversions, IDs versus
  indices, ordering, segment preservation/explicit concatenation,
  copy/view/mutation, fitted-state errors, exception categories,
  temporal/interval rules, RNG determinism, dtype/value-range/device coercion,
  accepted protocols, partial reads, cache behavior, eager/lazy materialization,
  and resource ownership where applicable.
- [ ] **4.8 Add result-schema checks.** Guard stable fields, result profiles,
  DataFrame identifiers, xarray dimensions/coordinates/data variables,
  identity/index and segment labels, fitted configuration/provenance, and
  serialization names selected by the contract.
- [ ] **4.9 Add metamorphic equivalence checks.** Test coordinate representation
  round trips, singular versus one-unit batch, full versus chunked/partial-read,
  arrays versus equivalent Pynapple inputs, and result fields versus
  DataFrame/xarray conversions where declared.
- [ ] **4.10 Add `tests/test_interoperability_contract.py`.** Exercise the defined
  tested cells—normally the declared minimum and locked/current versions—for core
  pandas schemas and optional Pynapple, pynwb/NWB, and xarray integrations. Test
  imports, direct inputs, representation/frame/unit and ID/index/segment
  preservation/loss, reversible schemas, partial reads, caching, dtype/scaling,
  lazy materialization, and file closure. Do not generate jobs for every
  intermediate release in an open-ended supported range.
- [ ] **4.11 Implement scale scenarios and reporting.** Add
  `scripts/benchmark_public_api.py` with fixed 1-, 100-, and 1,000-unit
  scenarios for applicable primary workflows. Track parameters, environment,
  segment/storage/chunk shape and alignment, peak memory, time, output
  allocation, dtype copies/scaling, partial versus full materialization,
  sparse/unit-specific preservation versus densification, and bounded-memory
  alternatives in `artifacts/SCALE_BASELINE.md`. Where no practical alternative
  exists, record the explicit limit and linked gap-report disposition. `--write`
  runs the scenarios;
  `--check` validates scenario/report structure without comparing
  machine-dependent raw timings.
- [ ] **4.12 Add robust regression gates.** Prefer structural allocation checks
  and stable workload thresholds over noisy single-run timing. Add a CI gate
  only after variance is characterized; benchmark reporting alone is useful and
  does not require a timing gate.
- [ ] **4.13 Make negative coverage durable.** Give structural validators
  checked-in negative fixtures and give semantic guards applicable failure-path
  assertions. For a guard family that cannot be exercised with a persistent
  negative fixture, deliberately mutate one representative fixture or
  implementation, run the narrow test, restore the mutation, and record the
  failing command plus actionable diagnostic in `artifacts/PROGRESS.md`. Do not
  require one-off manual mutation of every individual assertion.

### Phase 4 acceptance

- [ ] Intentional contract updates have a documented review path.
- [ ] Accidental symbol, signature, schema, optional-import, and dense-allocation
  regressions are detected.
- [ ] Accidental behavioral, identity, metadata, time-support, copy/mutation,
  coercion, lazy-evaluation, and resource-lifetime regressions are detected.
- [ ] Promised partial reads fail if they materialize the complete source;
  sparse/unit-specific conversions expose densification and allocation changes.
- [ ] Structural snapshot regeneration cannot bypass affected semantic tests.
- [ ] Defined core and optional integration test cells execute preservation/loss
  assertions, not only successful imports.
- [ ] Scale reports cover 1, 100, and 1,000 units for applicable workflows.

## Phase 5: Eligible cleanup and maintenance

**Dependencies:** accepted per-change decisions and completed compatibility
conditions. An agent must not interpret reaching this phase as blanket removal
authority.

- [ ] **5.1 Identify eligible hard-deprecated aliases.** Use curated metadata
  conditions rather than a date or named release. Soft-deprecated and permanent
  aliases are not eligible merely because time has passed.
- [ ] **5.2 Remove approved paths with migration documentation.** Keep each
  removal traceable to policy and evidence.
- [ ] **5.3 Review experimental APIs.** Promote, continue, or remove based on
  project-maintainer review (or an explicit owner override), demonstrated use,
  documentation, contract/interoperability coverage, scale characterization,
  and performance limits.
- [ ] **5.4 Review soft deprecations.** Confirm they remain safe, tested,
  documented, and correctly excluded from current teaching material; any move
  to hard deprecation starts a separately approved compatibility process.
- [ ] **5.5 Refresh artifacts.** Update inventory, design grammar and exceptions,
  result profiles, task map, snapshots, integration/loss matrix, deprecation
  ledger, semantic conformance, and scale baselines.

### Phase 5 acceptance

- [ ] Every removal has completed compatibility conditions.
- [ ] No soft-deprecated or permanent alias was removed through age alone.
- [ ] Published support tiers, migrations, and scale limits match runtime state.
- [ ] Published grammar, result profiles, interoperability versions, and loss
  declarations match runtime state.
- [ ] Cleanup timing was chosen independently of this plan.

## Phase 6: Closeout and follow-on gap report

- [ ] **6.1 Consolidate `artifacts/GAP_REPORT.md`.** Record proposed new
  facades, session QC, configuration/provenance objects,
  richer lossless interoperability containers, Array API/backend support, or
  bounded-memory algorithms in a gap report.
- [ ] **6.2 Complete proposal metadata.** Include repository or downstream
  evidence, affected journeys, scientific semantics, preservation/loss
  requirements, scale impact, and a recommendation for each proposal.
- [ ] **6.3 Route material additions.** For each accepted material proposal,
  create or link a separate design/scientific-review plan before implementation.
  If none is accepted, record that outcome; do not create speculative projects.

### Phase 6 acceptance

- [ ] Every out-of-scope finding has a disposition and evidence.
- [ ] No material addition was implemented through the curation plan.
- [ ] `artifacts/PROGRESS.md` identifies the next executable plan or states that
  curation is complete.

## Expected Validation Commands

The implementation must make these exact commands valid. Add focused test paths
earlier as their behavior is introduced; do not wait until Phase 4 to begin
testing the metadata schema or audit script. Fast structural and public-interface
checks run before dependency integration, scale, and the complete suite.

```bash
uv run python scripts/audit_public_api.py --check
uv run pytest -n 0 tests/test_public_api_contract.py
uv run pytest -n 0 tests/public_api/test_researcher_workflows.py
uv run pytest -n 0 tests/test_api_semantics.py
uv run pytest -n 0 tests/test_interoperability_contract.py
uv run python scripts/benchmark_public_api.py --check
uv run python scripts/test_doc_snippets.py
uv run mkdocs build --strict
uv run ruff check src/neurospatial scripts tests
uv run ruff format --check src/neurospatial scripts tests
```
