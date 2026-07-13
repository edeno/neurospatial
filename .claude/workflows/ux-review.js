export const meta = {
  name: 'ux-review',
  description: 'Ground-level UX evaluation of neurospatial through the ux-reviewer framework (interface usability, error messages, output formatting, workflow friction, accessibility) grounded in first-run usability probes, synthesized into UX-REVIEW.md',
  whenToUse: 'Evaluate day-to-day USABILITY of the public surface from a neuroscientist user perspective: are calls obvious, do errors say what/why/how, is output readable, can a first-timer succeed. Complements design-review.js (which owns the higher-altitude API/design view) — this is the call-site, first-run, error-recovery lens. Audits the CURRENT source; does not reuse prior review artifacts.',
  phases: [
    { title: 'Map', detail: 'map the UX-relevant surface: entry points, result formatting, error/warn sites, plotting' },
    { title: 'Probes', detail: 'three first-run usability probes (first field, mistake recovery, result interpretation)' },
    { title: 'Audit', detail: 'five ux-reviewer dimension audits grounded in the map + probes' },
    { title: 'Synthesize', detail: 'write UX-REVIEW.md in the ux-reviewer output format' },
  ],
}

// ---------------------------------------------------------------------------
// No args — evaluates the whole public UX surface. Run by scriptPath.
//
// Complementary to design-review.js: that workflow owns the higher-altitude
// API/design axes (mental model, composability, ecosystem fit). THIS one is
// the ground-level ux-reviewer lens — call-site ergonomics, error-message
// quality, output readability, first-run friction, accessibility of plots.
// Cross-reference design-review; do NOT re-derive its axes.
// ---------------------------------------------------------------------------

// Self-throttle so the fan-out stays within usage limits (>=4-5 concurrent
// agents trips the rate limiter). Mirrors design-review.js.
const CHUNK = 4
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
async function runChunked(items, fn, chunk = CHUNK) {
  const out = []
  for (let i = 0; i < items.length; i += chunk) {
    const part = await parallel(items.slice(i, i + chunk).map((it, j) => () => fn(it, i + j)))
    out.push(...part)
  }
  return out
}

// ---------------------------------------------------------------------------
// Shared context — the UX lens + the staleness guard
// ---------------------------------------------------------------------------
const STALENESS_GUARD = `SOURCE OF TRUTH = the CURRENT working tree under src/neurospatial/. The repo is mid-development (a feature branch is active), so any pre-existing review artifact (DESIGN-REVIEW.md, UX-REVIEW.md, .claude/reviews/) may be STALE and partly WRONG. Read the actual current code, signatures, error strings, and docstrings yourself. You MAY glance at a prior review only as a lead to check — never quote or reuse a finding without re-verifying it against the code as it stands now.`

const PROJECT_CONTEXT = `neurospatial is a Python library for discretizing continuous spatial environments into bins/nodes and analyzing neural/spatial data (place fields, Bayesian decoding, egocentric frames, object-vector & spatial-view cells, PSTH/events, animation, NWB I/O). Source under src/neurospatial/. Users are NEUROSCIENTISTS with varying technical backgrounds (wet-lab to computational), often in time-sensitive workflows, low tolerance for data-loss, and who frequently do NOT read docs first.

Established conventions (from CLAUDE.md) — treat these as the intended UX contract, and judge whether the code actually delivers it:
- Factory-only Environment construction (from_samples/from_graph/from_polygon/from_pixel_mask/from_grid_mask/from_polar_egocentric); bin_size/pixel_size REQUIRED; no bare Environment(); @check_fitted guards unfitted use.
- Egocentric angles animal-centered (0=ahead, +pi/2=left, -pi/2=right); allocentric 0=East.
- Canonical arg order (encoding: env, spike_times, times, positions, headings, object_positions, *, params; egocentric: positions, headings, targets; directional fns omit env).
- Encoding functions return RESULT OBJECTS (SpatialRateResult etc.) with terminal verbs: to_dataframe() (dense tidy), summary_table() (one row/unit), to_xarray(), summary() (dict), plot(ax=None). Cell-type predicates is_<celltype>_cell(...).
- Regions immutable (env.regions.update_region); is_linearized_track guards to_linear(); NumPy docstrings; units carried on env.units.

YOUR LENS = ux-reviewer.md GROUND LEVEL: is the call obvious at the call site, does every error say WHAT went wrong / WHY / HOW to fix, is result output readable and scannable, can a first-timer succeed without the manual, are defaults sane, are destructive ops (file overwrite) safe, are plot/animation defaults colorblind-safe. This is COMPLEMENTARY to design-review.js — do NOT re-litigate mental-model / composability / ecosystem-fit; stay at the usability altitude.

${STALENESS_GUARD}`

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------
const SURFACE_MAP_SCHEMA = {
  type: 'object',
  properties: {
    entry_points: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          module: { type: 'string' },
          public: { type: 'array', items: { type: 'string' } },
          result_classes: { type: 'array', items: { type: 'string' } },
        },
        required: ['module', 'public'],
      },
    },
    result_formatting: {
      type: 'array',
      description: 'result classes and which formatting/output methods they actually implement',
      items: {
        type: 'object',
        properties: {
          result_class: { type: 'string' },
          methods: { type: 'array', items: { type: 'string' } },
        },
        required: ['result_class', 'methods'],
      },
    },
    error_hotspots: {
      type: 'array',
      description: 'modules/files with dense raise/warnings.warn usage a user is likely to hit',
      items: {
        type: 'object',
        properties: {
          where: { type: 'string' },
          note: { type: 'string' },
        },
        required: ['where', 'note'],
      },
    },
    plotting_entries: { type: 'array', items: { type: 'string' } },
    golden_paths: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['entry_points', 'result_formatting', 'error_hotspots'],
}

const PROBE_SCHEMA = {
  type: 'object',
  properties: {
    probe: { type: 'string' },
    first_run_rating: { type: 'string', enum: ['smooth', 'workable', 'painful', 'blocked'] },
    sketch: { type: 'string', description: 'the actual code a user would write, verbatim' },
    friction_points: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          dimension: {
            type: 'string',
            enum: ['interface', 'errors', 'output', 'friction', 'accessibility'],
          },
          title: { type: 'string' },
          where: { type: 'string', description: 'function/file (path:line if known) the friction occurs at' },
          detail: { type: 'string' },
        },
        required: ['severity', 'dimension', 'title', 'where', 'detail'],
      },
    },
    error_recovery: {
      type: 'string',
      description: 'when the user hit a wall, did the error/warning tell them WHAT/WHY/HOW? quote the real message.',
    },
    output_interpretation: {
      type: 'string',
      description: 'could the user read and understand the result object output? what did repr/summary show?',
    },
    wins: { type: 'array', items: { type: 'string' } },
  },
  required: ['probe', 'first_run_rating', 'friction_points', 'error_recovery', 'output_interpretation'],
}

// Mirrors the ux-reviewer.md output sections exactly, so the ux-reviewer
// agentType's mandated format aligns with the structured return.
const DIMENSION_SCHEMA = {
  type: 'object',
  properties: {
    dimension: { type: 'string' },
    rating: { type: 'string', enum: ['USER_READY', 'NEEDS_POLISH', 'CONFUSING'] },
    critical_issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          where: { type: 'string' },
          impact: { type: 'string', description: 'concrete user impact' },
          fix: { type: 'string', description: 'specific, actionable fix (code-level where possible)' },
        },
        required: ['title', 'where', 'impact', 'fix'],
      },
    },
    confusion_points: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          where: { type: 'string' },
          why: { type: 'string' },
        },
        required: ['title', 'where', 'why'],
      },
    },
    suggested_improvements: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          where: { type: 'string' },
          benefit: { type: 'string' },
        },
        required: ['title', 'benefit'],
      },
    },
    good_patterns: {
      type: 'array',
      items: {
        type: 'object',
        properties: { title: { type: 'string' }, detail: { type: 'string' } },
        required: ['title', 'detail'],
      },
    },
  },
  required: ['dimension', 'rating', 'critical_issues', 'confusion_points', 'suggested_improvements', 'good_patterns'],
}

// ---------------------------------------------------------------------------
// Three FIRST-RUN usability probes (deliberately distinct from design-review's
// end-to-end journeys — these hunt usability friction, not composability).
// ---------------------------------------------------------------------------
const PROBES = [
  {
    key: 'first-place-field',
    title: 'Zero-knowledge first-timer computes their first place field',
    task: `A neuroscientist who has never used neurospatial has: spike_times (1 unit), times, and 2D positions as plain numpy arrays. With NO prior knowledge, they try to get a single place field to look at. Trace exactly what they must discover on their own: that Environment() is forbidden and which factory to call, that bin_size is required, that smoothing/occupancy defaults exist, how a result is returned and how to view it. Rate the FIRST-RUN experience. Quote the real errors they hit before success and judge whether those errors coach them forward.`,
  },
  {
    key: 'mistake-recovery',
    title: 'User makes the common mistakes and tries to recover',
    task: `A hurried user makes the mistakes real users make, one at a time: (1) calls a method on an unfitted/bare Environment; (2) picks a bin_size far too large or too small (triggering "No active bins" or a huge-grid warning); (3) calls to_linear() on a 2D environment; (4) forgets frame_times when animating; (5) passes positions with the wrong shape/units. For EACH, read the actual raised exception / warning string in the current source and grade it on WHAT went wrong / WHY / HOW to fix — does it name the offending argument and give a concrete next step, or just fail? This probe is specifically about error-message and recovery-path quality.`,
  },
  {
    key: 'result-interpretation',
    title: 'User inspects and interprets a result object',
    task: `A user has a SpatialRatesResult (population) and a single SpatialRateResult and wants to understand what they got WITHOUT reading docs: type it at the REPL (what does __repr__/print show?), call summary()/summary_table()/to_dataframe(), find the peak firing location and spatial information, and plot it. Judge OUTPUT FORMATTING: is the repr informative or bare, are units human-readable and labeled, does summary_table give the "one row per unit" a 1000-unit user wants, is the plot readable/colorblind-safe by default, and is it obvious which accessor to reach for. Read the actual result-class code.`,
  },
]

// ---------------------------------------------------------------------------
// Five ux-reviewer dimensions (run with agentType 'ux-reviewer').
// ---------------------------------------------------------------------------
const DIMENSIONS = [
  {
    key: 'interface',
    title: 'Interface Usability',
    focus: `Call-site ergonomics of the public functions and factories. Are parameter names domain-appropriate and unambiguous (e.g. is it obvious what bandwidth/min_occupancy/fill_value do and their units)? Are required arguments discoverable before a failure, or only via an exception? Is the factory-method pattern (no bare Environment()) a help or a tax at first contact? Is @check_fitted guidance surfaced to the user? Would a neuroscientist guess the right call without reading source?`,
  },
  {
    key: 'errors',
    title: 'Error Messages',
    focus: `Audit the actual raise/warnings.warn sites a user is likely to hit (start from the map's error_hotspots and the mistake-recovery probe). Grade each against WHAT went wrong / WHY it happened / HOW to fix it. Do messages name the offending argument and give a concrete corrective action (e.g. "bin_size=10.0 produced 0 active bins; try a smaller bin_size or bin_count_threshold=1")? Is jargon avoided or explained? Is the tone coaching, not blaming? Flag any silent failure or swallowed error where the user gets a wrong result with no signal.`,
  },
  {
    key: 'output',
    title: 'Output Formatting',
    focus: `Result objects and what the user sees. Is __repr__ informative (shape, n_units, key metric) or bare? Are units human-readable and labeled (cm, Hz, s) rather than raw floats? Does summary()/summary_table()/to_dataframe() present scannable, correctly-shaped tables (one row per unit for summary_table)? Is success/return explicit? Do NaN/low-occupancy bins and fill_value semantics read clearly? Ground this in the result-interpretation probe and the actual result-class source.`,
  },
  {
    key: 'friction',
    title: 'Workflow Friction',
    focus: `Minimal-steps-for-common-tasks, sensible defaults (do smoothing_method/bandwidth/min_occupancy defaults work for 80% without tuning?), progressive disclosure (advanced knobs not overwhelming beginners), first-run experience, and SAFETY on destructive ops — does to_file() overwrite silently? does clear_cache() before parallel rendering create a footgun? Are there undiscoverable required steps (frame_times, heading computation, occupancy thresholds) standing between a user and a result? Ground in the first-place-field probe.`,
  },
  {
    key: 'accessibility',
    title: 'Accessibility (translated for a scientific library)',
    focus: `Translate the web accessibility dimension to this library: are default colormaps in plot()/animate_fields()/napari colorblind-safe (avoid jet/rainbow; prefer viridis/cividis)? Does any output rely on color ALONE to convey meaning? Is terminal/repr output readable without color? For the napari viewer and video export, is there keyboard/non-mouse operability and clear progress feedback on long renders? Do NOT invent web-only concerns (ARIA) that don't apply — assess what actually reaches a user here.`,
  },
]

// ---------------------------------------------------------------------------
// Phase 1 — Map the UX-relevant surface
// ---------------------------------------------------------------------------
phase('Map')
const surface = await withRetry(() =>
  agent(
    `${PROJECT_CONTEXT}

Map the USER-FACING surface that a UX audit needs. Read src/neurospatial/__init__.py and each subpackage __init__, the result-class definitions, and grep for raise/warnings.warn. Also skim .claude/QUICKSTART.md and .claude/API_REFERENCE.md for the advertised golden paths (but verify against code).

Produce:
- entry_points: per module, the PUBLIC functions/classes a user calls, and the result classes returned.
- result_formatting: for each result class, which output methods it ACTUALLY implements (__repr__, summary, summary_table, to_dataframe, to_xarray, plot). Note missing ones.
- error_hotspots: modules/files with dense raise/warnings.warn a first-time user is likely to trip (factories, binning, decoding, animation).
- plotting_entries: the plot/animation entry points (plot(), animate_fields, napari).
- golden_paths: the canonical workflows the docs advertise.
Keep to the public surface; be complete but concise. Remember: verify against CURRENT source, not stale docs/reviews.`,
    { label: 'ux-surface-map', phase: 'Map', schema: SURFACE_MAP_SCHEMA },
  ),
)

const surfaceText =
  'ENTRY POINTS:\n' +
  (surface.entry_points || [])
    .map(
      (e) =>
        `- ${e.module}: ${(e.public || []).join(', ')}` +
        (e.result_classes && e.result_classes.length ? `  [results: ${e.result_classes.join(', ')}]` : ''),
    )
    .join('\n') +
  '\n\nRESULT FORMATTING (implemented output methods):\n' +
  (surface.result_formatting || []).map((r) => `- ${r.result_class}: ${(r.methods || []).join(', ') || '(none)'}`).join('\n') +
  '\n\nERROR HOTSPOTS:\n' +
  (surface.error_hotspots || []).map((h) => `- ${h.where}: ${h.note}`).join('\n') +
  (surface.plotting_entries && surface.plotting_entries.length
    ? `\n\nPLOTTING ENTRIES: ${surface.plotting_entries.join(', ')}`
    : '') +
  (surface.golden_paths && surface.golden_paths.length
    ? `\n\nADVERTISED GOLDEN PATHS:\n${surface.golden_paths.map((g) => '- ' + g).join('\n')}`
    : '')

log(`Mapped ${(surface.entry_points || []).length} modules, ${(surface.result_formatting || []).length} result classes, ${(surface.error_hotspots || []).length} error hotspots. Running ${PROBES.length} usability probes.`)

// ---------------------------------------------------------------------------
// Phase 2 — First-run usability probes (1 chunk of 3)
// ---------------------------------------------------------------------------
phase('Probes')
const probeResults = (
  await runChunked(PROBES, (p) =>
    withRetry(() =>
      agent(
        `${PROJECT_CONTEXT}

UX SURFACE MAP:
${surfaceText}

You are a NEUROSCIENTIST attempting this as a first-time user, reading ONLY the current public source to find your way. Do not assume anything the code does not spell out.

PROBE — ${p.title}:
${p.task}

Actually trace it against the current code (Read/Grep the real signatures, docstrings, and the exact raise/warning strings). Write the realistic code in 'sketch'. Log each friction point tagged by ux dimension (interface/errors/output/friction/accessibility) and severity. Fill error_recovery with the REAL messages encountered and whether they gave WHAT/WHY/HOW. Fill output_interpretation with what the user actually sees. Note genuine wins. Be concrete and fair — usability critique, not a bug hunt.`,
        { label: `probe:${p.key}`, phase: 'Probes', model: 'sonnet', schema: PROBE_SCHEMA },
      ),
    ).catch((e) => ({
      probe: p.title,
      first_run_rating: 'blocked',
      friction_points: [],
      error_recovery: `(probe errored: ${String((e && e.message) || e)})`,
      output_interpretation: '',
      wins: [],
      _error: true,
    })),
  )
).filter(Boolean)

const probeDigest = probeResults
  .map(
    (r) =>
      `### ${r.probe} [first-run: ${r.first_run_rating}]\n` +
      `Friction: ${(r.friction_points || []).map((f) => `(${f.severity}/${f.dimension}) ${f.title} @ ${f.where}`).join('; ') || 'none'}\n` +
      `Error recovery: ${r.error_recovery || '—'}\n` +
      `Output interpretation: ${r.output_interpretation || '—'}\n` +
      `Wins: ${(r.wins || []).join('; ') || '—'}`,
  )
  .join('\n\n')

log(`Ran ${probeResults.length} probes. Auditing ${DIMENSIONS.length} ux-reviewer dimensions (chunked at ${CHUNK}).`)

// ---------------------------------------------------------------------------
// Phase 3 — Five ux-reviewer dimension audits (chunks of 4 -> 4+1)
// ---------------------------------------------------------------------------
phase('Audit')
const dimensionResults = (
  await runChunked(DIMENSIONS, (d) =>
    withRetry(() =>
      agent(
        `${PROJECT_CONTEXT}

UX SURFACE MAP:
${surfaceText}

FIRST-RUN PROBE EVIDENCE (neuroscientists attempting real first-run tasks):
${probeDigest}

Evaluate ONE ux dimension of neurospatial for neuroscience users, using your ux-reviewer framework.

DIMENSION — ${d.title}:
${d.focus}

Read what you need from the CURRENT source to verify (do not trust stale docs/reviews). Assess THIS dimension only. Return your findings in the four ux-reviewer buckets — critical_issues, confusion_points, suggested_improvements, good_patterns — each grounded in a specific function/file/message (where), with concrete user impact and an actionable fix. Assign the dimension rating (USER_READY | NEEDS_POLISH | CONFUSING). Credit genuinely good UX; do not inflate severity.`,
        { label: `dim:${d.key}`, phase: 'Audit', agentType: 'ux-reviewer', schema: DIMENSION_SCHEMA },
      ),
    ).catch((e) => ({
      dimension: d.title,
      rating: 'NEEDS_POLISH',
      critical_issues: [],
      confusion_points: [],
      suggested_improvements: [],
      good_patterns: [],
      _error: String((e && e.message) || e),
    })),
  )
).filter(Boolean)

const erroredDims = dimensionResults.filter((d) => d._error).map((d) => d.dimension)
if (erroredDims.length) log(`WARNING: ${erroredDims.length} dimension audit(s) errored: ${erroredDims.join(', ')}`)

// ---------------------------------------------------------------------------
// Phase 4 — Synthesize UX-REVIEW.md in the ux-reviewer output format
// ---------------------------------------------------------------------------
phase('Synthesize')
const report = await withRetry(() =>
  agent(
    `You are writing the consolidated UX review of the neurospatial library for neuroscience users, in the ux-reviewer.md house style. Audience: the library's maintainer. This is the GROUND-LEVEL usability review; a separate design-review owns higher-altitude API/design — do not duplicate it.

DIMENSION AUDITS (JSON, each with its own rating):
${JSON.stringify(dimensionResults, null, 2)}

FIRST-RUN PROBES (JSON):
${JSON.stringify(
  probeResults.map((r) => ({
    probe: r.probe,
    first_run_rating: r.first_run_rating,
    sketch: r.sketch,
    friction_points: r.friction_points,
    error_recovery: r.error_recovery,
    output_interpretation: r.output_interpretation,
    wins: r.wins,
  })),
  null,
  2,
)}

Write a Markdown document with EXACTLY these sections:

# neurospatial — UX Review (usability, for neuroscience users)

## Overall Assessment
Lead with the overall rating on its own line: \`Rating: USER_READY | NEEDS_POLISH | CONFUSING\` (holistic, not merely the worst dimension), then 3-6 sentences justifying it: the biggest usability strengths and the most damaging friction. Then a small table of the five dimensions with each dimension's rating.

## First-run experience
One short paragraph per probe: its first-run rating and a 1-2 sentence narrative of where a newcomer flows and where they snag, citing the real error/output they hit. This grounds everything below.

## Critical UX Issues
The deduplicated, cross-dimension list of issues that will block or seriously frustrate users (data-loss risk, silent wrong results, complete confusion). Each: a bold title, the where-reference, the user impact, and the concrete fix. Merge duplicates raised by multiple dimensions and note when a theme is cross-cutting.

## Confusion Points
What will confuse users and why (deduplicated), with where-references. Lower stakes than Critical.

## Suggested Improvements
Prioritized under **High / Medium / Low**. Each: bold title, concrete change, one-line user-impact rationale. This is the action list — make it specific and implementable.

## Good UX Patterns (keep)
The genuinely good usability decisions to preserve through any refactor.

Rules: reference real functions/files/messages. Balance praise and critique honestly. Do not invent findings beyond the JSON. If a dimension errored, say so under Overall Assessment rather than fabricating its content. Keep the ux-reviewer rating vocabulary exactly (USER_READY | NEEDS_POLISH | CONFUSING).`,
    { label: 'ux-synthesize', phase: 'Synthesize' },
  ),
)

return report
