export const meta = {
  name: 'design-review',
  description: 'Holistic API & design evaluation of neurospatial for neuroscience users — user-journey walkthroughs + 7 cross-cutting design axes synthesized into DESIGN-REVIEW.md',
  whenToUse: 'Evaluate the public API/design quality (mental model, onboarding, consistency, discoverability, domain fit, ecosystem fit, composability) from the neuroscientist user perspective, grounded in concrete end-to-end journeys. Reuses the per-module review signal in .claude/reviews/.',
  phases: [
    { title: 'Map', detail: 'enumerate the public API surface' },
    { title: 'Journeys', detail: 'attempt real end-to-end tasks as a user, log friction' },
    { title: 'Axes', detail: 'evaluate 7 cross-cutting design axes' },
    { title: 'Synthesize', detail: 'write DESIGN-REVIEW.md' },
  ],
}

// ---------------------------------------------------------------------------
// No args needed — this evaluates the whole public API. Run by scriptPath.
// ---------------------------------------------------------------------------
const REVIEWS_DIR = '.claude/reviews'

// Self-throttle so a fan-out doesn't trip the API rate limiter (see module-review).
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

const PROJECT_CONTEXT = `neurospatial is a Python library for discretizing continuous spatial environments into bins/nodes and analyzing neural/spatial data (place fields, decoding, egocentric frames, object-vector & spatial-view cells, animation, NWB I/O). Source under src/neurospatial/. Users are neuroscientists.
Established conventions (from CLAUDE.md): factory-only Environment construction (from_samples/from_graph/from_polygon/from_pixel_mask/from_grid_mask/from_polar_egocentric; bin_size/pixel_size required); egocentric angles animal-centered (0=ahead,+pi/2=left), allocentric 0=East; canonical arg order (encoding: env, spike_times, times, positions, headings, object_positions, *, params; egocentric: positions, headings, targets; directional fns omit env); encoding functions return result objects (e.g. SpatialRateResult); regions immutable; @check_fitted for fitted state; is_linearized_track guards to_linear; NumPy docstrings.
IMPORTANT: a per-module CODE review already exists in ${REVIEWS_DIR}/ (5 batch reports + SUMMARY.md) containing api-consistency / ux-ergonomics / docs-consistency findings. READ the relevant ones and REUSE that signal — do not re-derive it. Your job is the HIGHER-ALTITUDE design view a per-module bug review cannot see.`

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------
const API_MAP_SCHEMA = {
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
    golden_paths: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['entry_points'],
}

const JOURNEY_SCHEMA = {
  type: 'object',
  properties: {
    journey: { type: 'string' },
    feasible: { type: 'string', enum: ['smooth', 'workable', 'painful', 'blocked'] },
    sketch: { type: 'string', description: 'brief outline of the code a user would actually write' },
    friction_points: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocker', 'major', 'minor'] },
          title: { type: 'string' },
          where: { type: 'string', description: 'API/function/file the friction occurs at' },
          detail: { type: 'string' },
        },
        required: ['severity', 'title', 'where', 'detail'],
      },
    },
    wins: { type: 'array', items: { type: 'string' } },
    missing_capabilities: { type: 'array', items: { type: 'string' } },
  },
  required: ['journey', 'feasible', 'friction_points'],
}

const AXIS_SCHEMA = {
  type: 'object',
  properties: {
    axis: { type: 'string' },
    summary: { type: 'string' },
    strengths: {
      type: 'array',
      items: {
        type: 'object',
        properties: { title: { type: 'string' }, detail: { type: 'string' } },
        required: ['title', 'detail'],
      },
    },
    weaknesses: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          title: { type: 'string' },
          where: { type: 'string' },
          detail: { type: 'string' },
        },
        required: ['severity', 'title', 'where', 'detail'],
      },
    },
    recommendations: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          priority: { type: 'string', enum: ['high', 'medium', 'low'] },
          title: { type: 'string' },
          change: { type: 'string', description: 'the concrete API/design change' },
          rationale: { type: 'string' },
        },
        required: ['priority', 'title', 'change', 'rationale'],
      },
    },
  },
  required: ['axis', 'summary', 'strengths', 'weaknesses', 'recommendations'],
}

// ---------------------------------------------------------------------------
// Concrete user journeys (real neuroscience end-to-end tasks)
// ---------------------------------------------------------------------------
const JOURNEYS = [
  {
    key: 'place-fields-from-nwb',
    title: 'Place fields from an NWB session, then decode + animate',
    task: `A user has an NWB file with position (SpatialSeries) and spike times for N units. They want to: load it, build an Environment, compute place fields for every unit, run Bayesian decoding of position, and animate the decoded posterior with the trajectory overlaid.`,
  },
  {
    key: 'linear-track-laps',
    title: 'Linear/W-track linearization → lap segmentation → directional place fields',
    task: `A user has a linear or W-track session. They want to linearize position to 1D, split into inbound/outbound laps (trials), and compute direction-specific place fields, then compare them.`,
  },
  {
    key: 'egocentric-object-vector',
    title: 'Egocentric frame → object-vector cell analysis',
    task: `A user has 2D position, head/body orientation, and object locations. They want heading (from velocity or body axis), egocentric bearing/distance to objects, an object-vector firing field, and a yes/no object-vector-cell classification.`,
  },
  {
    key: 'simulate-decode-reactivation',
    title: 'Simulate a session → encode → decode → reactivation/assembly stats',
    task: `A user wants synthetic ground truth: simulate a session (place cells + trajectory), fit encoding models, decode, and run population reactivation/assembly statistics — checking that one stage's outputs feed the next without manual glue.`,
  },
  {
    key: 'event-aligned',
    title: 'Event-aligned analysis (reward/ripple) → PSTH → GLM regressors',
    task: `A user has event times (rewards or ripples) and spikes. They want a peri-event histogram, GLM regressors built from the events, and to relate event-locked firing to the animal's position.`,
  },
]

// ---------------------------------------------------------------------------
// The 7 cross-cutting design axes
// ---------------------------------------------------------------------------
const AXES = [
  {
    key: 'mental-model',
    title: 'Mental model & core abstractions',
    focus: `Is Environment the right central object, and are layout/regions/encoding/decoding a coherent, minimal set? Does the factory-method pattern (no bare Environment()) clarify or tax users? Is there conceptual integrity, or are there overlapping/competing abstractions? Does the bin/node/graph model match how neuroscientists think about space?`,
  },
  {
    key: 'onboarding',
    title: 'Onboarding & the golden path',
    focus: `How far is "import to first real result" for a newcomer? Are there sensible defaults, a clear happy path, and good first-run errors? How much undiscoverable required setup (e.g. frame_times, heading computation, occupancy thresholds) stands between a user and a place field? Lean on the journey reports for concrete friction.`,
  },
  {
    key: 'api-consistency',
    title: 'Cross-module API consistency & predictability',
    focus: `As a WHOLE: argument order, parameter naming, units conventions, keyword-only separators, and especially RESULT-OBJECT PARITY (do sibling functions return analogous result classes with analogous accessors?). Aggregate the api-consistency findings already in ${REVIEWS_DIR}/ into systemic patterns rather than per-site bugs.`,
  },
  {
    key: 'discoverability',
    title: 'Discoverability & namespacing',
    focus: `Top-level and submodule __init__ exports, namespacing depth, can a user find the right function for a task? Are docstrings/Examples usable as navigation? Is there a coherent "where does X live" story? Note import-path friction and any functions that are powerful but effectively hidden.`,
  },
  {
    key: 'domain-fit',
    title: 'Domain fit & vocabulary',
    focus: `Does the API speak neuroscience (place cells, occupancy, theta phase, ripples/SWRs, egocentric/allocentric, spatial information)? Do function names and arguments match field conventions? What standard analyses are missing or awkward that users will expect? Where does terminology diverge from the literature?`,
  },
  {
    key: 'ecosystem-fit',
    title: 'Ecosystem fit & interoperability',
    focus: `How well does it interoperate with the data objects neuroscientists already hold — NWB, plain numpy arrays, pandas/xarray? Are inputs/outputs idiomatic (arrays vs custom objects) and easy to get data into/out of? Evaluate on its own terms (no peer-library benchmark): is adoption low-friction for someone arriving with NWB + numpy?`,
  },
  {
    key: 'composability',
    title: 'Composability across modules',
    focus: `Do one module's outputs feed the next cleanly (encoding result objects → decoders; segmentation → encoding; simulation → encoding → decoding)? Where are the seams that force manual array-wrangling/glue code? Is there a smooth pipeline, or islands? Use the journeys to locate the seams.`,
  },
]

// ---------------------------------------------------------------------------
// Phase 1 — Map the public API surface
// ---------------------------------------------------------------------------
phase('Map')
const apiMap = await withRetry(() =>
  agent(
    `${PROJECT_CONTEXT}

Map the PUBLIC API surface of neurospatial so design reviewers know what users actually touch. Read the top-level package __init__ (src/neurospatial/__init__.py) and each subpackage's __init__ exports, plus .claude/QUICKSTART.md and .claude/API_REFERENCE.md.

For each module, list the PUBLIC entry points (exported functions/classes a user calls) and the result classes it returns. Also list the "golden paths" the docs advertise (the canonical workflows). Keep it to the public surface — skip private helpers. Be reasonably complete but concise.`,
    { label: 'api-map', phase: 'Map', schema: API_MAP_SCHEMA },
  ),
)

const apiMapText =
  (apiMap.entry_points || [])
    .map(
      (e) =>
        `- ${e.module}: ${(e.public || []).join(', ')}` +
        (e.result_classes && e.result_classes.length ? `  [results: ${e.result_classes.join(', ')}]` : ''),
    )
    .join('\n') +
  (apiMap.golden_paths && apiMap.golden_paths.length
    ? `\n\nDocumented golden paths:\n${apiMap.golden_paths.map((g) => '- ' + g).join('\n')}`
    : '')

log(`Mapped ${(apiMap.entry_points || []).length} modules' public API. Walking ${JOURNEYS.length} journeys, then ${AXES.length} design axes.`)

// ---------------------------------------------------------------------------
// Phase 2 — User journeys (as a real user, log friction)
// ---------------------------------------------------------------------------
phase('Journeys')
const journeyResults = (
  await runChunked(JOURNEYS, (j) =>
    withRetry(() =>
      agent(
        `${PROJECT_CONTEXT}

PUBLIC API SURFACE:
${apiMapText}

You are a NEUROSCIENTIST using neurospatial for the first time, attempting this end-to-end task using ONLY the public API:

TASK — ${j.title}:
${j.task}

Actually trace it: read the real functions/signatures/docstrings (Read/Grep) and the QUICKSTART, and check ${REVIEWS_DIR}/ for known rough edges. Write the realistic code a competent user would write (in 'sketch'), and log every FRICTION point: missing function, confusing/dangerous argument order, unclear or implicit units, a required step that is hard to discover (e.g. you must compute frame_times / headings / occupancy thresholds yourself), excessive boilerplate, or a footgun where the easy call silently does the wrong thing. Rate overall feasibility. Note genuine WINS (things that were smooth) and any MISSING capabilities a user would expect for this task. Be concrete and fair — this is a design critique, not a bug hunt.`,
        { label: `journey:${j.key}`, phase: 'Journeys', schema: JOURNEY_SCHEMA },
      ),
    ).catch((e) => ({ journey: j.title, feasible: 'blocked', friction_points: [], _error: String((e && e.message) || e) })),
  )
).filter(Boolean)

const journeyDigest = journeyResults
  .map(
    (r) =>
      `### ${r.journey} [${r.feasible}]\n` +
      `Friction: ${(r.friction_points || []).map((f) => `(${f.severity}) ${f.title} @ ${f.where}`).join('; ') || 'none'}\n` +
      `Wins: ${(r.wins || []).join('; ') || '—'}\n` +
      `Missing: ${(r.missing_capabilities || []).join('; ') || '—'}`,
  )
  .join('\n\n')

log(`Walked ${journeyResults.length} journeys. Evaluating ${AXES.length} design axes.`)

// ---------------------------------------------------------------------------
// Phase 3 — Design axes (each reuses reviews + journey friction)
// ---------------------------------------------------------------------------
phase('Axes')
const axisResults = (
  await runChunked(AXES, (a) =>
    withRetry(() =>
      agent(
        `${PROJECT_CONTEXT}

PUBLIC API SURFACE:
${apiMapText}

USER-JOURNEY FRICTION (from neuroscientists attempting real end-to-end tasks):
${journeyDigest}

You are evaluating the DESIGN of neurospatial along ONE axis for neuroscience users.

AXIS — ${a.title}:
${a.focus}

Read what you need (public API, QUICKSTART/CLAUDE.md, and the existing per-module findings in ${REVIEWS_DIR}/ — reuse their api-consistency/ux/docs signal rather than re-deriving it). Assess THIS axis only. Give: a short summary judgement; concrete strengths (what's genuinely well-designed); weaknesses (severity-tagged, with where); and prioritized, concrete recommendations (actual API/design changes, not vague advice). Ground claims in specific functions/modules. Be fair: credit good design, and don't inflate severity.`,
        { label: `axis:${a.key}`, phase: 'Axes', schema: AXIS_SCHEMA },
      ),
    ).catch((e) => ({
      axis: a.title,
      summary: `(evaluation errored: ${String((e && e.message) || e)})`,
      strengths: [],
      weaknesses: [],
      recommendations: [],
      _error: true,
    })),
  )
).filter(Boolean)

const erroredAxes = axisResults.filter((a) => a._error).map((a) => a.axis)
if (erroredAxes.length) log(`WARNING: ${erroredAxes.length} axis evaluation(s) errored: ${erroredAxes.join(', ')}`)

// ---------------------------------------------------------------------------
// Phase 4 — Synthesize DESIGN-REVIEW.md
// ---------------------------------------------------------------------------
phase('Synthesize')
const report = await withRetry(() =>
  agent(
    `You are writing the consolidated design & API evaluation of the neurospatial library for neuroscience users. Audience: the library's maintainer.

AXIS EVALUATIONS (JSON):
${JSON.stringify(axisResults, null, 2)}

USER JOURNEYS (JSON):
${JSON.stringify(
  journeyResults.map((r) => ({
    journey: r.journey,
    feasible: r.feasible,
    sketch: r.sketch,
    friction_points: r.friction_points,
    wins: r.wins,
    missing_capabilities: r.missing_capabilities,
  })),
  null,
  2,
)}

Write a Markdown document with EXACTLY these sections:

# neurospatial — API & Design Review (for neuroscience users)

## Executive summary
3-6 sentences: the overall design verdict, the 2-3 biggest strengths, and the 2-3 most important design problems. Be direct and balanced.

## User journeys
One short subsection per journey: its feasibility rating, a 1-2 sentence narrative of where it flows and where it snags, and the key friction points (with the API/where). This grounds the rest of the document.

## Design axes
One subsection per axis (use the axis titles). For each: a 2-4 sentence judgement, a short **Strengths** list, and a short **Weaknesses** list (keep the [severity] tag and the where-reference). Be concise; do not dump every item.

## Prioritized recommendations
A single ranked, DEDUPLICATED list of concrete design/API changes drawn from all axes and journeys, grouped under **High**, **Medium**, **Low**. Each item: a bold title, the concrete change, and a one-line rationale tying it to user impact. Merge recommendations that several axes raised (note when a theme is cross-cutting). This is the action list — make it specific and implementable.

## What's working (keep)
A short list of design decisions that are good and should be preserved, so a refactor doesn't break them.

Rules: be specific and reference real functions/modules. Balance praise and critique honestly. Do not invent findings beyond the JSON above. If an axis errored, note it under Executive summary rather than fabricating its content.`,
    { label: 'synthesize', phase: 'Synthesize' },
  ),
)

return report
