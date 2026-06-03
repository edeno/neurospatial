export const meta = {
  name: 'module-review',
  description: 'Large 11-dimension review of named neurospatial module(s) — fans out by (module x dimension) for a scientific Python library used by neuroscientists',
  whenToUse: 'Audit one or more modules (e.g. ["encoding","decoding"]) for correctness, stats, units, reproducibility, tests, API/UX, docs, and performance. args = list of module names; each module is reviewed by its own set of scoped agents.',
  phases: [
    { title: 'Discover', detail: 'resolve module name(s) to per-module source/test/doc files' },
    { title: 'Review', detail: '11 specialist finders per module read scoped code in parallel' },
    { title: 'Verify', detail: 'adversarial skeptic refutes false positives per finding' },
    { title: 'Synthesize', detail: 'per-module report sections + executive summary' },
  ],
}

// ---------------------------------------------------------------------------
// args: a module name or list of names, e.g. ["encoding","decoding"] or "stats".
// Each module is resolved to its own files and reviewed by its own agents so no
// single finder has to hold the whole tree in context.
// ---------------------------------------------------------------------------
const modules = Array.isArray(args) ? args : args ? [args] : []
if (!modules.length) {
  throw new Error(
    'module-review requires args: a list of module names, e.g. {name:"module-review", args:["encoding","decoding"]}',
  )
}

// How many (module x dimension) tasks to run per chunk. Kept below the runtime
// concurrency cap on purpose: a large simultaneous fan-out trips the API's
// transient rate limiter. Lower this if you still see rate-limit errors.
const CHUNK = 5

// Retry transient rate-limit / overload errors (NOT logic errors). Without a
// sleep primitive the retry is immediate, but within a small chunk the burst is
// small enough that a retry usually lands after the spike clears.
const RATE_LIMIT_RE = /rate.?limit|temporarily limiting|overloaded|throttl|\b429\b|\b529\b/i
async function withRetry(fn, attempts = 3) {
  let lastErr
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn()
    } catch (e) {
      lastErr = e
      if (!RATE_LIMIT_RE.test(String((e && e.message) || e))) throw e
    }
  }
  throw lastErr
}

// ---------------------------------------------------------------------------
// Structured-output schemas
// ---------------------------------------------------------------------------
const MANIFEST_SCHEMA = {
  type: 'object',
  properties: {
    modules: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          module: { type: 'string' },
          exists: { type: 'boolean' },
          source_files: { type: 'array', items: { type: 'string' } },
          test_files: { type: 'array', items: { type: 'string' } },
          doc_files: { type: 'array', items: { type: 'string' } },
        },
        required: ['module', 'source_files', 'test_files', 'doc_files'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['modules'],
}

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['critical', 'important', 'suggestion'] },
          title: { type: 'string' },
          file: { type: 'string' },
          line: { type: 'string', description: 'line number or range, e.g. "120" or "120-135", or "N/A"' },
          rationale: { type: 'string' },
          suggested_fix: { type: 'string' },
        },
        required: ['severity', 'title', 'file', 'line', 'rationale', 'suggested_fix'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    is_real: { type: 'boolean' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
    reason: { type: 'string' },
  },
  required: ['is_real', 'confidence', 'reason'],
}

// ---------------------------------------------------------------------------
// Shared project context — keeps every agent grounded in the conventions
// without re-deriving them from CLAUDE.md each time.
// ---------------------------------------------------------------------------
const PROJECT_CONTEXT = `This is the neurospatial library (repo root = current working directory; source under src/neurospatial/).
Key project conventions (from CLAUDE.md):
- Construction is via factory methods (Environment.from_samples / from_graph / from_polygon / from_pixel_mask / from_grid_mask / ...); never bare Environment(). Grid-inferring factories require an explicit bin_size (from_pixel_mask requires pixel_size).
- Egocentric angles are animal-centered: 0=ahead, +pi/2=left, -pi/2=right. Allocentric: 0=East, +pi/2=North. These two frames must never be silently mixed.
- Canonical argument order — encoding fns: (env, spike_times, times, positions, headings, object_positions, *, params). Egocentric ops: (positions, headings, targets). Documented exception: directional fns take (spike_times, times, headings, *, ...) with NO env.
- Naming: use positions (not trajectory) and position_bins (not trajectory_bins).
- Regions are immutable (update only via env.regions.update_region).
- Fitted-state methods use the @check_fitted decorator; 1D-only methods (to_linear) are guarded by is_linearized_track.
- Encoding functions return result objects (e.g. SpatialRateResult), not bare arrays.
- NumPy-style docstrings only.
This is a SCIENTIFIC library used by neuroscientists: the worst failure mode is a plausible-but-wrong numerical result that silently ends up in a paper. Weight correctness accordingly.`

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------
function manifestTextFor(m) {
  return [
    `Module: ${m.module}`,
    `Source files:\n${(m.source_files || []).map((f) => '  ' + f).join('\n') || '  (none)'}`,
    `Test files:\n${(m.test_files || []).map((f) => '  ' + f).join('\n') || '  (none)'}`,
    `Doc files:\n${(m.doc_files || []).map((f) => '  ' + f).join('\n') || '  (none)'}`,
  ].join('\n\n')
}

function finderPrompt(focusTitle, focusBody, manifestText) {
  return `You are a specialist code reviewer for the neurospatial scientific Python library.
${PROJECT_CONTEXT}

FILES IN SCOPE — review ONLY these (use Read/Grep/Glob to read the real code before judging — never guess):
${manifestText}

YOUR FOCUS — ${focusTitle}:
${focusBody}

Rules:
- Only report an issue you can tie to a specific file and line/range AFTER reading the code there.
- Stay inside your focus area AND inside the listed files; other reviewers and other modules are covered separately. Do not duplicate their scope.
- Severity: "critical" = wrong results / crash / data corruption; "important" = real bug or convention violation that should be fixed; "suggestion" = improvement.
- If you find nothing in your focus, return an empty findings list. Do NOT invent issues to look productive.
Return your findings via the structured output tool.`
}

function verifyPrompt(dimensionKey, f) {
  return `You are an adversarial verifier on a code review of the neurospatial scientific Python library. A "${dimensionKey}" reviewer reported this finding:

Title: ${f.title}
Severity: ${f.severity}
Location: ${f.file}:${f.line}
Rationale: ${f.rationale}
Proposed fix: ${f.suggested_fix}

Read the ACTUAL code at that location and its surrounding context (Read/Grep) before deciding — do not rely on the rationale alone.

Your job is to kill false positives. Mark is_real=false ONLY if the finding clearly misreads the code, is not actually a problem, is already handled nearby, or falls outside the stated focus. When you are genuinely uncertain, mark is_real=true — for scientific code we prefer a human triaging a real-looking finding over silently dropping a correctness bug.

Give a one- to three-sentence reason citing what you actually saw in the code.`
}

function moduleSynthesisPrompt(m, coverageRows, found) {
  return `You are writing the review section for ONE module of the neurospatial library: ${m.module}.
These findings are already adversarially verified. Deduplicate findings that point at the same file:line and the same underlying issue (keep the clearest; if several dimensions flagged it, list the dimensions together).

PER-DIMENSION COVERAGE for this module (JSON). For each row: "confirmed" = findings that
SURVIVED adversarial verification (these are what appear below); "found" = total findings
before verification (so found>confirmed means the skeptic refuted some); found = -1 means the
dimension errored and did not run.
${JSON.stringify(coverageRows, null, 2)}

CONFIRMED FINDINGS (JSON):
${JSON.stringify(found, null, 2)}

Produce a Markdown section EXACTLY in this shape:

## \`${m.module}\`

### Critical (must fix before use)
- [dimension] **Title** — \`file:line\`. One line: why it is wrong + the fix.

### Important (should fix)
(same format)

### Suggestions (nice to have)
(same format)

### Strengths
- Brief, specific notes on what is well done (infer from dimensions whose "confirmed" count is 0 — i.e. nothing survived verification, whether or not the finder initially raised something).

Rules: every issue KEEPS its \`file:line\` reference and its [dimension] tag. If a subsection is empty, write "_None found._". Be concise and concrete. Do NOT invent findings not in the JSON above.`
}

function rollupPrompt(allConfirmedSlim, coverage) {
  return `You are writing the EXECUTIVE SUMMARY at the top of a multi-module review report for the neurospatial library.
Modules reviewed: ${modules.join(', ')}.

ALL CONFIRMED FINDINGS (JSON, already verified; detail lives in the per-module sections below):
${JSON.stringify(allConfirmedSlim, null, 2)}

Write a concise executive summary. Start at "## Executive Summary" (do NOT emit a top-level # heading):
- One line: count of findings by severity.
- **Top issues** — the 3-8 highest-impact items across all modules (severity first), each: [module / dimension] **Title** — \`file:line\`.
- **Cross-cutting patterns** — any issue type recurring across modules (e.g. the same convention violated in several places), if any. Omit if none.
- **Recommended order** — what to fix first and why.
Be specific and short. Do not restate every finding.`
}

// ---------------------------------------------------------------------------
// The 11 review dimensions
// ---------------------------------------------------------------------------
const DIMENSIONS = [
  {
    key: 'algorithmic-correctness',
    title: 'Algorithmic / array-logic correctness (NOT units, stats, or NaN-handling — those have their own reviewers)',
    body: `- Indexing, masking, slicing, broadcasting, reshape/transpose, and axis= arguments.
- Off-by-one / fencepost errors in binning, edges vs centers, half-open vs closed intervals.
- Vectorized rewrites that are NOT equivalent to the obvious loop they replace.
- Incorrect accumulation or normalization (e.g. rate = spikes / occupancy, occupancy normalization).
- Graph / neighbor / geodesic logic and coordinate<->bin mapping correctness.
- Wraparound handling for circular quantities (e.g. bearing wrap at +-pi).`,
  },
  {
    key: 'numerical-robustness',
    title: 'Numerical robustness and degenerate inputs',
    body: `- NaN / Inf propagation: handled/masked, or silently poisoning sums/means/argmax?
- Empty arrays, single-sample, all-identical-value, zero-variance inputs.
- Zero-occupancy bins, division by zero, log(0), 0/0, sqrt of negatives.
- Underflow/overflow in exp/log (Bayesian posteriors, softmax) — is log-space or normalization used?
- Float equality (==) where a tolerance is required.`,
  },
  {
    key: 'statistical-validity',
    title: 'Statistical validity',
    body: `- Circular statistics: correct mean / resultant length / variance, Rayleigh & V-test, weighting.
- Shuffle / surrogate / permutation controls: correct null, sufficient shuffles, preserved structure, no leakage.
- Spatial-information & selectivity metrics: occupancy weighting and small-sample bias correction.
- Multiple-comparison handling when many bins/cells are tested.
- p-value computation: one- vs two-sided, rank off-by-one, (+1) corrections.`,
  },
  {
    key: 'units-and-frames',
    title: 'Units and reference-frame consistency',
    body: `- Mixed/implicit units (cm vs m, s vs ms, Hz, rad vs deg) passed between functions without conversion.
- bin_size, bandwidth, min_speed, distances: units documented and consistent across call sites?
- Egocentric (0=ahead) vs allocentric (0=East) applied consistently; no silent frame mixups.
- Angle convention/range consistency: is the *frame-correct* range used ([-pi, pi) vs [0, 2pi)) consistently across call sites? (The wrap *arithmetic* itself is algorithmic-correctness's job — focus here on the choice of range/convention.)
- dt derived correctly from times (no diff off-by-one, assumes uniform sampling without checking?).`,
  },
  {
    key: 'reproducibility-rng',
    title: 'Reproducibility and RNG hygiene',
    body: `- np.random.Generator / passed-in rng or seed vs the legacy GLOBAL np.random state.
- Functions with randomness expose a seed/rng parameter and are deterministic given it.
- Independent random streams where independence is required (per-neuron simulation, per-shuffle); no accidentally shared/correlated state.
- No reseeding inside loops; no nondeterministic iteration order (set/dict) feeding numerical output.`,
  },
  {
    key: 'tests',
    agentType: 'pr-review-toolkit:pr-test-analyzer',
    title: 'Test quality and coverage (NOTE: there is no PR/diff — review the listed test + source files directly)',
    body: `- Behavioral coverage of the public functions in scope; identify critical untested paths.
- Edge cases exercised: empty / NaN / degenerate inputs, boundary bins, single sample.
- Tests that assert real behavior vs trivial smoke tests or mock-only tests (see testing-anti-patterns).
- Missing regression tests for this library's failure mode (wrong numbers, not just crashes).
- file = the test file, or the source file lacking coverage; line where applicable.`,
  },
  {
    key: 'silent-failure',
    agentType: 'pr-review-toolkit:silent-failure-hunter',
    title: 'Silent failures and input validation (NOTE: no diff — review the listed files directly)',
    body: `- try/except that swallows errors; bare except; returning None/NaN/empty on error without signaling.
- Silent dtype/shape coercion; fallback defaults that mask bad input.
- Missing validation of array-length agreement (spike_times / times / positions), input shape, and value ranges.
- For this library, silently producing a wrong-but-plausible array is the worst outcome — flag it as critical.`,
  },
  {
    key: 'api-consistency',
    title: 'Public API consistency with project conventions',
    body: `- Canonical argument order; env-first for encoding, animal-state-before-targets for egocentric; the documented directional exception respected.
- Construction only via factory methods; required bin_size / pixel_size present.
- Encoding functions return result objects (e.g. SpatialRateResult), not bare arrays; consistent result-class shape across siblings.
- Keyword-only separator (*) before algorithm params.
- Naming: positions / position_bins (not trajectory / *_bins); parameter names consistent across sibling functions.
- @check_fitted on fitted-state methods; is_linearized_track guard before to_linear; regions treated as immutable.
Report inconsistencies vs sibling functions and vs these conventions.`,
  },
  {
    key: 'ux-ergonomics',
    title: 'API / developer ergonomics for the neuroscientist user (this is API UX, NOT visual UI)',
    body: `- Discoverability: are the right entry points exported and easy to find? Sensible defaults so common cases need few args?
- Error messages: actionable, naming the offending value and a fix hint (CLAUDE.md's rich "No active bins" error is the bar)?
- Result objects: convenient accessors/metrics, or do they force manual array gymnastics?
- Docstring Examples that copy-paste and run (imports present, runnable).
- Naming clarity; units stated in the signature/docstring.
- Footguns: easy-to-misuse argument orders, silent unit assumptions, required-but-easy-to-forget args.`,
  },
  {
    key: 'docs-consistency',
    title: 'Documentation consistency (docstrings + .claude docs vs actual code)',
    body: `- Docstring parameters, defaults, types, and return types match the real signature and code.
- NumPy docstring format (Parameters / Returns / Examples sections present and well-formed).
- Examples are importable and correct (function names, argument order, return shapes match the CURRENT API).
- Claims in .claude/*.md (CLAUDE.md, QUICKSTART, API_REFERENCE) match the implemented behavior/signatures of the in-scope module.
- Stated units and conventions in docs match the code.
- Stale references: renamed/removed functions, outdated defaults, wrong return descriptions.`,
  },
  {
    key: 'performance-memory',
    title: 'Performance and memory scalability (datasets: long recordings, many neurons, fine bins)',
    body: `- Python loops over samples / neurons / bins that should be vectorized.
- O(n^2) or worse: full pairwise distance matrices, repeated recomputation that could be cached.
- Large dense allocations (full grids, dense graphs) where sparse/chunked would scale; unnecessary array copies.
- Loop-invariant quantities recomputed inside loops.
- Only report cases likely to matter at realistic scale; state the scaling and a concrete fix. Do not micro-optimize cheap code.`,
  },
]

// ---------------------------------------------------------------------------
// Phase 1 — Discover (one agent enumerates files for every named module)
// ---------------------------------------------------------------------------
phase('Discover')
const manifest = await agent(
  `You are scoping a code review for the neurospatial library (repo root = current working directory).

Resolve these module name(s) to actual files, one entry per module: ${modules.join(', ')}

For each module, enumerate (use Glob/Grep — do not review anything yet):
- source files: src/neurospatial/<module>/**/*.py (exclude __pycache__).
- test files: tests covering it — Glob tests/**/*<module>*.py AND grep tests/ for imports of the module.
- doc files: .claude/*.md (and docs/, README) that reference the module by name (Grep).

Set exists=false for any module name that does not resolve to a directory under src/neurospatial/. Return repo-relative paths.`,
  { label: 'discover', phase: 'Discover', schema: MANIFEST_SCHEMA },
)

const seenModules = new Set()
const moduleEntries = (manifest.modules || [])
  .filter((m) => m && m.exists !== false)
  // Dedupe by name so a manifest that lists a module twice doesn't double-fan-out
  // and render two sections under one key.
  .filter((m) => (seenModules.has(m.module) ? false : seenModules.add(m.module)))
const skipped = (manifest.modules || []).filter((m) => m && m.exists === false).map((m) => m.module)
if (skipped.length) log(`Skipped (not found under src/neurospatial/): ${skipped.join(', ')}`)
if (!moduleEntries.length) {
  return `# Module Review\n\n_No requested module resolved to source files. Requested: ${modules.join(', ')}._`
}
if (manifest.notes) log(`Discovery notes: ${manifest.notes}`)

// ---------------------------------------------------------------------------
// Phase 2 — Review -> Verify, fanned out over (module x dimension).
// pipeline() means each (module,dimension) verifies as soon as its finder
// returns; a slow finder never blocks a fast one's verification.
// ---------------------------------------------------------------------------
const tasks = []
for (const m of moduleEntries) {
  for (const d of DIMENSIONS) tasks.push({ m, d })
}
log(
  `Reviewing ${tasks.length} (module x dimension) tasks in chunks of ${CHUNK} to stay under API rate limits.`,
)

// Stage 1: find, scoped to ONE module's files. Retry transient rate limits, then
// catch so one failure doesn't drop silently — it becomes a recorded _error.
const findStage = (t) => {
  const opts = { label: `find:${t.m.module}:${t.d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA }
  if (t.d.agentType) opts.agentType = t.d.agentType
  return withRetry(() => agent(finderPrompt(t.d.title, t.d.body, manifestTextFor(t.m)), opts)).catch((e) => ({
    findings: [],
    _error: String((e && e.message) || e),
  }))
}

// Stage 2: adversarially verify each finding from this (module, dimension).
// Guard against a null/malformed `review`: a THROW here would drop the whole
// (module,dimension) to null and lose its error signal, so we coerce + wrap.
const verifyStage = (review, t) => {
  const findings = Array.isArray(review && review.findings) ? review.findings : []
  return parallel(
    findings.map((f) => () =>
      withRetry(() =>
        agent(verifyPrompt(t.d.key, f), {
          label: `verify:${t.m.module}:${t.d.key}`,
          phase: 'Verify',
          schema: VERDICT_SCHEMA,
        }),
      )
        .then((v) => ({ ...f, module: t.m.module, dimension: t.d.key, verdict: v }))
        .catch(() => ({
          ...f,
          module: t.m.module,
          dimension: t.d.key,
          verdict: { is_real: true, confidence: 'low', reason: 'verifier errored; kept conservatively' },
        })),
    ),
  )
    .then((verified) => ({
      module: t.m.module,
      dimension: t.d.key,
      error: (review && review._error) || null,
      verified: verified.filter(Boolean),
    }))
    .catch((e) => ({
      module: t.m.module,
      dimension: t.d.key,
      error: (review && review._error) || `verify stage errored: ${String((e && e.message) || e)}`,
      verified: [],
    }))
}

// Self-throttle: process tasks in small sequential chunks so in-flight load stays
// well under the runtime concurrency cap. A 143-task burst trips the API's
// transient rate limiter; chunking keeps the sustained request rate safe.
const taskResults = []
for (let i = 0; i < tasks.length; i += CHUNK) {
  const part = await pipeline(tasks.slice(i, i + CHUNK), findStage, verifyStage)
  taskResults.push(...part)
  const errs = taskResults.filter((r) => r && r.error).length
  log(`Reviewed ${Math.min(i + CHUNK, tasks.length)}/${tasks.length} tasks (${errs} errored so far).`)
}

// ---------------------------------------------------------------------------
// Aggregate per module
// ---------------------------------------------------------------------------
const coverage = taskResults.filter(Boolean).map((r) => ({
  module: r.module,
  dimension: r.dimension,
  found: r.error ? -1 : r.verified.length,
  confirmed: r.error ? 0 : r.verified.filter((f) => f.verdict && f.verdict.is_real).length,
  ...(r.error ? { note: `finder error: ${r.error}` } : {}),
}))

const confirmedByModule = {}
for (const r of taskResults.filter(Boolean)) {
  const keep = r.verified.filter((f) => f.verdict && f.verdict.is_real)
  ;(confirmedByModule[r.module] = confirmedByModule[r.module] || []).push(...keep)
}
const allConfirmed = Object.values(confirmedByModule).flat()
log(`Confirmed ${allConfirmed.length} finding(s) after verification across ${moduleEntries.length} module(s).`)

// ---------------------------------------------------------------------------
// Phase 3 — Synthesize: one section per module (agent only where there are
// findings; clean modules get a cheap JS stub), then an executive summary.
// ---------------------------------------------------------------------------
phase('Synthesize')

function moduleStub(m, cov) {
  const errored = cov.filter((c) => c.found === -1).map((c) => c.dimension)
  const files = (m.source_files || []).length + (m.test_files || []).length + (m.doc_files || []).length
  const allErrored = cov.length > 0 && errored.length === cov.length
  const head = allErrored
    ? `## \`${m.module}\`\n\n⚠️ **NOT REVIEWED** — all ${cov.length} dimensions errored. This is **not** a clean result.`
    : `## \`${m.module}\`\n\n_No issues survived verification._`
  return (
    `${head}\n\n` +
    `**Coverage:** ${cov.length - errored.length}/${cov.length} dimension(s) completed on ${files} file(s)` +
    (errored.length ? `; errored: ${errored.join(', ')}` : '') +
    '.'
  )
}

const sections = await parallel(
  moduleEntries.map((m) => () => {
    const found = confirmedByModule[m.module] || []
    const cov = coverage.filter((c) => c.module === m.module)
    if (!found.length) return Promise.resolve(moduleStub(m, cov))
    return agent(moduleSynthesisPrompt(m, cov, found), { label: `report:${m.module}`, phase: 'Synthesize' })
  }),
)

// Loud failure surfacing: a review that quietly reports "no issues" when its
// finders actually crashed is the exact silent-failure footgun this tool exists
// to catch. If any (module x dimension) errored, lead with a banner.
const erroredRows = coverage.filter((c) => c.found === -1)
const erroredModules = [...new Set(erroredRows.map((c) => c.module))]
const ranCount = coverage.length - erroredRows.length
const errorBanner = erroredRows.length
  ? `> ⚠️ **INCOMPLETE REVIEW — ${erroredRows.length} of ${coverage.length} (module × dimension) checks FAILED to run** (e.g. API rate limiting) and were NOT reviewed.\n` +
    `> Absence of findings below does **not** mean those areas are clean. Re-run the affected modules in smaller batches.\n` +
    `> Affected modules: ${erroredModules.join(', ')}.\n\n`
  : ''

let header
if (!allConfirmed.length) {
  const body =
    erroredRows.length === coverage.length
      ? `**Every dimension errored — nothing was actually reviewed.** See the warning above.`
      : `_No issues survived verification across the ${ranCount} dimension run(s) that completed (of ${coverage.length})._`
  header = `# Module Review: ${modules.join(', ')}\n\n${errorBanner}${body}`
} else {
  const slim = allConfirmed.map((f) => ({
    severity: f.severity,
    title: f.title,
    module: f.module,
    dimension: f.dimension,
    file: f.file,
    line: f.line,
  }))
  const exec = await agent(rollupPrompt(slim, coverage), { label: 'exec-summary', phase: 'Synthesize' })
  header = `# Module Review: ${modules.join(', ')}\n\n${errorBanner}${exec}`
}

return [header, ...sections.filter(Boolean)].join('\n\n---\n\n')
