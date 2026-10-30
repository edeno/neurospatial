# Concrete next steps toward the project vision

**Created:** 2026-10-29
**Status:** Proposed delivery sequence; implementation has not started under this
sequence. Existing delivered work remains recorded in [TASKS.md](TASKS.md).

The next milestone is a complete, small experiment connecting a changing spatial
environment to the animal's sensory access, possible movement, observed behavior,
and neural activity. Follow it with exploration history and ordered routes.
Each step should produce usable public operations and a worked example.

This document sequences the remaining work in [PLAN.md](PLAN.md) against the
[project vision](../../vision.md). Existing phase files remain the detailed
contracts, subject to checking their descriptions against current code. New
public API names and the
access-sampling interface below are proposals requiring normal design review.

## 1. Select the environment that applies at each time

**First implementation change:** a state-interval selector with a small,
hand-checkable changing-barrier fixture and its tests.

Use a two-room planar environment with a target in the second room. Start with a
transparent divider, then open a doorway at a declared time. Add an opaque
divider as a control case. Keep cell geometry, ordering, units, and coordinate
frame consistent while allowing walls and movement connections to differ.
Verify correspondence explicitly; equal array lengths do not establish it.

Accept explicit half-open intervals, state identifiers, and a mapping to existing
immutable environments. Select the new state at an exact shared boundary. Reject
overlaps, unknown identifiers, and incompatible geometry or metadata. Uncovered
time remains unknown. Reuse environment values and existing interval utilities;
ordinary arrays and functions remain sufficient to use the feature.

**Done when:** exact change times, uncovered times, and invalid schedules have
known expected answers; chunked and whole-recording selection agree; and saving
the environment files plus an explicit interval table and identifier mapping
reproduces the selection. The first example can use existing environment storage
and a documented table recipe without defining a new general session format.

**Builds on:** [phase 5 state selection](phase-5-event-alignment-and-features.md),
[shared state contracts](shared-contracts.md#features-and-environmental-conditions),
and [phase 2](phase-2-environment-contracts.md).

## 2. Measure visual access and movement access through those states

Compose the selector with existing visibility, spatial feature, and navigation
operations. Produce sample-aligned outputs that identify the input sample,
time, state, and target, together with:

- Visibility under the declared field of view and physical wall model.
- Bearing and distance to the target where defined.
- Whether the movement graph provides a path to the target, and its length.
- Separate validity for each quantity, with reasons for unknown values.
- The assumptions and settings used for the calculation.

Missing heading can make visibility unknown while movement access remains
defined. An unknown state makes state-dependent quantities unknown. A known
disconnected target is unreachable; this must remain distinguishable from
missing information. Derived interpolation and event alignment must respect
state boundaries and the existing observation-support rules.

The initial movement model is a point moving through the environment's graph.
Document that assumption. Body clearance, reaching, contact, and manipulation
need separately specified models before they can be claimed as measured access.
An egocentric feature query alone does not establish visibility.

**Done when:** the transparent divider permits sight and blocks movement, the
opaque divider blocks both, and the open doorway permits both at a selected
position and heading. Include a reachable target hidden around a corner and
missing-heading/state cases. Establish expected answers independently of the
functions being checked.

Deliver a worked example using the same measurements to identify approaches,
compare neural rates across conditions, and display behavior and sensory access.
Use existing conditioned-rate and event-alignment tools. Show occupancy and
speed/heading differences; comparisons are descriptive unless an appropriate
inferential procedure is supplied. A synthetic recording should include a known
unchanged-cell control and a deliberately changed response, with independently
seeded behavior and spikes. Include explicit recording gaps.

Use state-specific animation scenes or clips with labeled change times so walls
are rendered correctly. A general animation of changing geometry can be a later
extension. Add instructions for replacing synthetic observations with a real
recording.

**Builds on:** `ops/visibility.py`, `ops/feature_sampling.py`,
`behavior/navigation.py`, [phase 5](phase-5-event-alignment-and-features.md), and
[phase 6](phase-6-neural-comparisons.md). The combined sampling interface is new
design work; reuse existing calculations and public result conventions.

## 3. Add exploration and prior-visit history

Deliver this in two small changes using
[phase 3](phase-3-exploration-and-crossings.md):

1. Observed crossing records, first observed visits, and coverage over time or
   observed distance. Require an independently defined eligible arena for the
   coverage denominator. Distinguish the fraction of bins visited from the
   fraction of physical area or volume visited; physical weighting needs
   validated measures.
2. Revisit records and features such as previous region, prior visit count, and
   time or observed distance since the previous visit. Calculate history before
   filtering event cohorts. Treat recording gaps and incomplete history
   explicitly, and use only preceding observations for predictive features.

Extend existing region visits and observation handling. Keep the current event's
identity and original sample indices. A first observed visit does not establish
that the animal has never visited the place outside the recorded observations.

**Done when:** a small hand-specified route has exact expected visits, crossing
counts, coverage, and history; dropped observations cannot invent visits or
travel; appending future data does not change earlier predictive features.

Deliver a free-foraging example with a tracking dropout, coverage and revisit
plots, and neural comparisons around observed entries. Export the event rows,
settings, and support needed to reproduce the analysis.

## 4. Add repeated-route analysis

Implement the first bounded slice of
[phase 4](phase-4-path-analysis.md): explicitly aligned trajectories, resampling
by physical distance, and ordered segment matches with direction checks. Retain
the source times and observed-run boundaries. Return the matched and surrounding
segments so the results can feed behavioral and neural comparisons.

**Done when:** the same route at different speeds matches; opposite-direction
travel is handled according to the explicit direction setting; one shared point
cannot count as a shared route; and gaps cannot create a continuous match.
Check small cases against an independent exhaustive calculation, then measure
runtime and memory on representative recordings.

Extend the behavioral example to compare the same route after different prior
visits. Label whole-recording matches as retrospective descriptions. Predictive
examples must construct route references and labels without using future or
held-out information.

## Validation and documentation accompany every step

Start a compact method-validation table with the first implementation change and
extend it as each workflow ships. For occupancy, path distance, visibility,
spatial rates, and conditioned comparisons, record the definition, implemented
variants, assumptions, independent numerical checks, simulation checks, and
known limits. Link existing evidence and identify missing evidence explicitly.
Compare implementations only after matching conventions, inputs, and settings.

Use public APIs in examples. Verify observation gaps, state changes, units,
spatial and neural identities, and export/reload behavior wherever relevant.
Record random seeds and versions. Run focused tests, required API checks, typing
and lint, and the affected executable documentation. Broad existing regression
checks complement the new scientific cases.

Correct stale dimensionality guidance and phase prerequisites in the relevant
documentation changes. An implemented function, a verified workflow, and a
validated scientific conclusion should have distinct evidence in the record.

## Follow-on work

Once the state and access rules work in the demonstrated workflows, add a small
agent adapter with explicit observations, allowed actions, time advancement,
rewards, and a reproducible reset. Validate its trajectories with the same
behavioral tools. Agent training systems remain separate consumers. This is a
follow-on proposal outside the currently accepted continuous-behavior scope.

Use concrete experiments to select the next interaction models, such as body
clearance or measured contact, and the next 3D sensory operations. Define their
required measurements and independent checks before extending the shared
interfaces. The existing continuous-behavior scope does not include a general
statistical framework; broader inferential methods need their own scoped plan.
