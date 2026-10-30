# Project Vision

**neurospatial aims to be a shared foundation for representing spatial
environments and studying behavior and neural activity within them.** Scientists
should be able to define an environment once, save and share it, measure what
happens in it, and use it across analysis, modeling, simulation, agent experiments,
and visualization.

The package should represent how the animal can sense and interact with its
surroundings, and analyze behavior in the context of those opportunities,
experimental conditions, and prior experience.

It should enable reproducible, statistically defensible comparisons of behavior
and neural activity across animals, sessions, and changing environments,
supported by transparent methods and community validation.

These are long-term goals that guide development. They do not imply that every
capability is available today or that every method applies in every dimension.
The user guides describe supported workflows; the
[design principles](contributing.md#design-principles) guide implementation.

## Goals

### 1. Describe and measure experimental environments

Provide a common way to parameterize the spaces used in spatial neuroscience:
their geometry, connectivity, boundaries, barriers, landmarks, and regions.
Support meaningful measurements of position, distance, direction, extent, and
accessible paths, with explicit coordinate frames and physical units.

Keep the physical environment, its numerical discretization, and the animal's
observations distinguishable. An unvisited location can still belong to the
environment, and an analysis boundary need not be a physical wall.

Represent changes in barriers, rewards, landmarks, and cues over time, with
explicit experimental conditions.

### 2. Represent the environment from the animal's perspective

Describe the sensory information available to an animal at a particular place
and time. Account for position, body and head orientation, measured gaze where
available, field of view, occlusion, and sensory range under explicit models.
Support both world coordinates and coordinates relative to the animal. Allow
sensory cues beyond vision to be represented through measurements and declared
models.

Represent opportunities for interaction: what the animal can approach, reach,
contact, traverse, or manipulate, given its capabilities and the current
environment. Sensory access and movement constraints can differ: a transparent
barrier may leave a reward visible while preventing access to it. Body size,
posture, object properties, and task conditions may affect which actions are
possible.

Modeled access is not observed behavior: visibility alone is not evidence that
a cue was perceived, and proximity alone is not evidence of contact.

### 3. Bring recordings in faithfully and keep environments durable

Read the outputs of acquisition, tracking, and spike-sorting tools, including
NWB files and pynapple objects, without losing their meaning. Keep timestamps,
units, recording gaps, and unit identities intact. Make tracking errors,
position jumps, and head direction derived from LEDs or pose visible and
correctable with explicit, documented steps rather than silent cleaning, so
that every later analysis knows which observations it can trust.

Provide a documented, versioned way to save spatial environments and exchange
them between tools and laboratories. Preserve the geometry, topology, spatial
identities, units, coordinate frames, annotations, and provenance needed to
interpret and reuse them.

Loading an environment should recover its scientific meaning and supported
operations without requiring the original tracking data or replaying its
construction. Compatibility and migration rules should make changes explicit.

Extend reproducibility to analyses: retain input references, data selections,
observation gaps and exclusions, parameters, software versions, and random seeds
needed to explain and reproduce a result.

### 4. Support complex spaces in 1D, 2D, and 3D

Make tracks, branching mazes, irregular arenas, connected spaces, and volumetric
environments usable through coherent spatial concepts. Measurements and analyses
should respect the environment's geometry and topology.

State each method's dimensional and geometric requirements. Generalize methods
where scientifically justified, and explain restrictions where their definitions
depend on a particular geometry or dimension.

### 5. Provide trustworthy, comparable neuroscience methods

Bring classic spatial neuroscience and behavioral analyses into a coherent
library for both neural activity and behavior. Scientists should be able to use
established methods, understand their assumptions, and compare alternatives on
the same environments and observations.

Aim to provide reference implementations that the field can adopt or use to
evaluate other implementations. Document method definitions, literature sources,
conventions, and variants. Test numerical behavior against analytic cases,
scientific invariants, independently established expectations, and appropriate
reference implementations, with justified tolerances. Explain differences caused
by scientific choices as well as implementation errors.

Support scientific inference with effect sizes, uncertainty estimates,
appropriate null models, and evaluation on held-out observations. Make the
assumptions about sampling, temporal dependence, and independent experimental
units explicit so scientists can judge what a result supports.

Read out what neural populations represent. Decode position and other
variables from spikes with calibrated uncertainty, and evaluate decoders on
held-out data. Detect and test non-local content, such as replay during
sharp-wave ripples, theta sequences, and the reactivation of assemblies during
rest, against appropriate null models.

Enable comparisons across animals, sessions, conditions, and environments.
Account for coordinate alignment, spatial correspondence, differences in
behavioral sampling, and the observations eligible for each comparison.

### 6. Enable richer behavioral analysis in spatial context

Support analysis across timescales, from movement and individual interactions
to action sequences, routes, exploration, decisions, and changes with experience.
Scientists should be able to measure approaches and retreats, encounters,
visits and revisits, route choices, and adaptation to changes in barriers,
rewards, or cues, in both trial-based tasks and continuous behavior.

Combine trajectories, measured pose or gaze, task events, interactions, and
behavioral annotations with explicit definitions and preserved timing and
observation gaps. Make these measurements composable into analyses of sequences,
behavioral states, and history. Accept externally produced annotations and model
outputs with their provenance and uncertainty.

Interpret behavior relative to available paths, sensory access, opportunities
for interaction, and prior observations and actions. Connect these measurements
to neural analyses so scientists can ask how activity at the same location
depends on what the animal could sense, what it was doing, or where it had been.

### 7. Provide reusable regressors and features for complex models

Turn spatial environments and observations into well-defined model inputs.
Provide regressors, covariates, and basis representations for generalized linear
models (GLMs), state-space models, and other statistical models, as well as
observation features for reinforcement learning (RL). Examples include position,
direction, distance to boundaries or targets, sensory access, interaction
opportunities, task conditions, and behavioral history.

Preserve timestamps, observation support, spatial identities, units, feature
definitions, and assumptions so users can combine these inputs with neural and
behavioral data. Make history requirements explicit. Features
intended for prediction should use information available at the prediction time,
with fitted transformations learned from training observations. For RL, make
the information available to the agent explicit and distinguish its observations
from environment information used only for evaluation or supervision.

Support use in external modeling and learning packages through documented arrays,
tables, and optional adapters. Users could fit models with NeMoS or another
library, or supply features to their own RL agents. Generating and using these
features must not require NeMoS or any particular modeling or RL framework.

### 8. Simulate behavior and neural activity with known ground truth

Generate trajectories, neural responses, and spikes on the same environments
used for recorded data. Support validation of classic methods, development of
new methods, and experiments that reproduce or explore known phenomena.

Use shared definitions of sensory access and interaction opportunities when
simulating observations, actions, and their relationship to neural responses.

Make simulation parameters, random seeds, and ground truth accessible. Use these
to assess recovery, bias, uncertainty, and failure cases. Include independent
checks so that a simulator and estimator sharing the same mistake cannot serve
as the sole evidence of correctness.

### 9. Let simulated agents act in the same environments

Allow agents to observe and move through the same spatial environments used to
analyze animals. Provide reusable spatial operations and interfaces for agent
simulators, with explicit movement constraints and environmental features.

Make agents' access to environmental information and available actions explicit,
so experiments can control what they can sense and interact with over time.

Keep agent policies and learning rules separate from the shared environment
representation. Agent trajectories and neural models should be measurable with
the same tools used for experimental data.

### 10. Make space and time easy to inspect together

Help spatial neuroscientists visualize environments, behavior, neural activity,
and analysis results in their spatial and temporal context. Support useful
static figures, synchronized exploration, comparisons, and shareable outputs.

Show modeled sensory access, interaction opportunities, and behavioral sequences
alongside the animal's movement and neural activity.

Plots and animations should expose units, spatial relationships, recording gaps,
and unsupported observations so scientists can interpret results and diagnose
problems.

### 11. Serve as a foundation for future methods

Provide reusable spatial representations, numerical operations, and analysis
interfaces that researchers can build on. New methods should be able to reuse
geometry, observation support, simulation, storage, and visualization while
remaining compatible with existing scientific Python workflows.

Success includes external packages using these foundations and comparing their
results with neurospatial. Adoption should be possible one component at a time.

Make validation an ongoing community process. Support contributed method
variants, shared reference datasets and benchmarks, reviewable validation
evidence, and documented discrepancies between implementations. Keep the scope
of each validation claim clear as methods and dependencies evolve.

## Not in scope

neurospatial works with the outputs of these steps rather than performing them:

- Spike sorting and other preprocessing of raw electrophysiology.
- Tracking animals in video and estimating pose.
- Fitting general statistical models, such as GLMs or state-space models; the
  package supplies their inputs (goal 7).
- Training agents: policies and learning rules belong to agent frameworks
  (goal 9).
- Managing data across experiments, such as databases and pipelines.

## How these goals guide development

The scientist's experience remains the highest design priority. Scientific
correctness, clear assumptions, consistent interfaces, useful errors, and
interpretable results are part of that experience.

Three rules apply to every goal:

- Keep the physical environment, its discretization, modeled sensory access and
  interaction opportunities, observed behavior, and inferred perception or
  knowledge distinguishable.
- Keep missing observations identifiable as unknown, never as zero.
- Label predictions and measurements with their assumptions and uncertainty,
  and label model-based interpretations as interpretations.

Runnable documentation, interoperability, figures that do not rely on color
alone, and practical memory and runtime on realistic recordings also apply
across all goals.

Evaluate proposed work by the scientific workflow it enables and the evidence
that the workflow preserves meaning. In particular:

- Can a scientist load a recording with its gaps, tracking errors, and unit
  identities visible, and correct them with explicit steps?
- Can a scientist describe an experimental space, save it, reload it, and obtain
  equivalent spatial measurements?
- Can a scientist distinguish what the animal could sense or interact with from
  what was observed, with explicit assumptions and missing information?
- Can routes, encounters, choices, and behavioral sequences be analyzed in the
  context of available actions, sensory cues, and prior experience, then related
  to neural activity?
- Can spatial and behavioral features become inputs to GLMs, state-space models,
  and RL agents while preserving their timing, meaning, and permitted information,
  without requiring a particular modeling or learning framework?
- Can recorded animals and simulated agents use the same environment and be
  compared through the same behavioral and neural analyses?
- Can a scientist decode what a population represents, including replay and
  other non-local content, with calibrated uncertainty and an appropriate null?
- Can competing methods be evaluated on matched observations with documented
  assumptions and independently checked numerical expectations?
- Can scientists assess uncertainty and generalization with methods appropriate
  to the experimental design and dependence between observations?
- Can a result be reproduced and compared across sessions or changing
  environments with its data selections, spatial alignment, and settings intact?
- Can a scientist inspect results in space and time, understand missing data,
  and reuse the components in a new method?
- Can contributors and external packages run shared benchmarks and inspect the
  evidence behind a method's validation claims?

Together, these goals define the project's direction. Individual implementation
plans specify scope, dependencies, validation, and delivery order.
