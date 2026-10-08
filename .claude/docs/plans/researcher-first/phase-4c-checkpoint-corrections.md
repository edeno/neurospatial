# Phase 4c — Researcher-workflow checkpoint corrections

**Requires:** Phase 4b merged. This phase was added by the 2026-10-07 checkpoint;
the gate remains held until these corrections merge and the journeys are repeated.

[← back to PLAN.md](PLAN.md) · [executing a phase](executing.md) ·
[overview](overview.md) ·
[checkpoint evidence](../../../../docs/reviews/RESEARCHER_WORKFLOW_CHECKPOINT_2026-10-07.md)

## Checkpoint additions

The first checkpoint completed every journey from public docs with zero source
reads, but rated the track journey painful and found misleading simulation and
statistical contracts. This phase fixes those bounded issues. Native 1D
plotting, richer output, and remaining composition recipes are assigned to
their existing later phases; they do not require a new facade or public time type.

Read `executing.md` before implementation. The checkpoint's measurements are
evidence, not target outputs to tune to. Follow the same regression-first,
CHANGELOG-per-commit, all-extras validation and independent-review workflow.

## Tasks

### 4c.1 The documented linear-track duration controls the returned recording

The public signature defines `duration` as the session duration in seconds.
The checkpoint's small reproducer is:

```python
from neurospatial.simulation import linear_track_session

for duration in (1.0, 2.0):
    sim = linear_track_session(
        duration=duration, track_length=1.0, bin_size=0.2,
        n_place_cells=2, n_laps=2, seed=42,
    )
    print(duration, sim.times[-1] - sim.times[0], len(sim.times),
          sim.metadata["duration"])
```

Measured before correction: both spans are `18.772000000000002` seconds and
both have 9,387 samples; metadata records 1.0 and 2.0 respectively. A larger
60-second request produced 4,483,393 samples. Do not hide this by rescaling
timestamps in a downstream recipe or substituting the requested value into
metadata without repairing the returned data.

- Find the `linear_track_session` → `simulate_session` → lap-trajectory path
  by public symbols. Write a regression first and record the unmodified failure.
- Make duration a real constraint on the returned trajectory and spike streams.
  Define and document how requested duration and lap count determine traversal
  timing; state any speed constraint or infeasible combination explicitly.
  Preserve requested laps rather than silently discarding them. If honoring
  that contract conflicts with another documented scientific default, document
  the conflict and ask the maintainer under `executing.md` before choosing a
  different contract.
- Keep positions/times aligned, timestamps strictly increasing, spikes on the
  same clock, units in cm/seconds, seeded reproducibility, and truthful metadata.
  Generated sample spans should match the requested duration to the documented
  sampling convention, within one sample period; metadata must state the same
  convention. Validate at least two requested durations, not one pinned count.
- Inspect related lap-based convenience calls for the same ignored-duration
  handoff; add a regression only where the same defect is demonstrated.
- Preserve gap-free outputs for unaffected OU/open-field generators and avoid
  unrelated changes to neural firing models or classification thresholds.

### 4c.2 Graph coordinates and temporal direction have separate roles

Notebook 05 states that the same spatial point gets different linear
coordinates/bins on opposite traversals, and that linear coordinates increase
monotonically during backtracking. The single-edge runtime disproves that:
`[25,50,75,50,25]` maps to those same coordinates and bins `[4,9,14,9,4]`.

- Correct `examples/05_track_linearization.py`, its notebook, and synchronized
  `docs/examples` copies. Search the whole tutorial for the repeated
  direction/history claims, plot titles, printed conclusions and takeaways.
- Explain graph coordinates as geometry. Do not generalize the single-edge
  observation into claims about all branch-disambiguation behavior.
- Add or link one complete graph-track recipe that explicitly segments
  inbound/outbound trials and supplies labels to `compute_directional_place_fields`.
  Notebook 21's public `segment_trials` → `goal_pair_direction_labels` recipe
  already works on the graph environment; reuse it rather than invent an API.
- Execute a real out-and-back fixture that demonstrates repeated positions
  share geometric bins while direction-conditioned maps recover different
  planted peaks. Assert each map's peak against its ground truth within one bin.
- Put the complete joined recipe under the executable-doc guard using a small,
  self-contained fixture and a marker, not a skipped symbolic fragment.

### 4c.3 Effect size and standardized activity do not establish significance

Public `assembly_activation` docs associate activation >2 with p<0.05.
A documented one-neuron pattern with 90 zero-count bins and 10 one-count bins
returns standardized activation values `[-1/3,3]`: 10% exceed 2. Its standard
deviation is correctly 1; the defect is probability interpretation, not output
normalization. `ExplainedVarianceResult` also calls EV >0.1 significant and
describes reversed EV as a plain reverse prediction, whereas its function's
controlled REV swaps the template/control roles.

- Correct the public activation and EV-result docstrings, including inference
  language in examples/notes and any matching public explanation. Standardized
  activity, a heuristic effect-size cutoff and a calibrated significance test
  must be named separately. Do not invent a p-value or significance API.
- Describe controlled REV consistently with `explained_variance_reactivation`;
  explain that without a baseline control EV and REV are equal by construction.
- Keep formulas, standardization, thresholds and numerical arrays unchanged.
  Retain the small runtime counterexamples as documentation evidence; test
  actual relevant numerical invariants rather than freezing prose strings.
- Explain that auto-selected significant dimensions can have no thresholded
  core members, and that synthetic/in-sample demonstrations do not establish
  biological assemblies or held-out decoding accuracy.

## Validation slice

| Check | Asserts |
| --- | --- |
| Duration regression | Two durations produce appropriate different time spans; metadata, trajectory and spikes agree; requested laps remain represented. |
| Unaffected simulation comparison | OU/open-field scientific arrays remain unchanged for seeded inputs. |
| Graph-direction recipe | Repeated coordinates share geometric bins; explicit direction labels produce two maps near their planted centers. |
| Assembly/EV counterexamples | Standardization remains correct; controlled/no-control EV behavior is numerically preserved. |
| Executable docs and notebook synchronization | Joined recipe runs with no disallowed skip; Python/notebook copies contain the same corrected explanation. |

Run the full definition of done in `executing.md`, including the documentation
suite and strict MkDocs build. After this phase merges, repeat the UX probes
and five design journeys using public documentation only, recording calls/LOC and any
source-reading fallbacks in a dated report. Do not claim quantitative reduction
against unavailable October code counts. Do not start Phase 5a unless the
repeated checkpoint is clearly easier and its blocking findings are closed.

## Review

Dispatch an independent reviewer before opening the implementation PR. Confirm
the duration defect and interpretation claims are corrected, numerical changes
are restricted to the demonstrated simulation defect, and the other scientific
defaults remain intact. Check that no private-helper imports, hard-coded
duration rescaling, mandatory bundle, API shim or inventory/count-freeze test
was introduced as a workflow repair.

## After merge

Record Phase 4c merged in `PLAN.md` with the repeated researcher-workflow
checkpoint as the next step. A first-review report or a merged correction PR
alone does not mark the checkpoint passed.
