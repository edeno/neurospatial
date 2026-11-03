# Phase 6 — Do encoding, behavior and figures give the right answer

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md)

Prerequisite: phase 5 merged (#281). Runs alongside the dataset-loaders plan
([../dataset-loaders/PLAN.md](../dataset-loaders/PLAN.md)), which owns the
loaders' file truth and the published-result reproductions.

## Why this phase

Today an operation is `ok` when the call does not raise. Only decoding has a
truth built in (the true position), so decoding error became the number every
batch compares. The survey calls 12 of the 175 public functions in
`encoding`, `behavior`, `events`, `regions` and `stats`, and no plotting or
animation code.

Goal: the encoding and behavior workflows a spatial neuroscientist runs are
checked against a known answer, on the real recordings' own paths and
apparatus, and the figures and animations show the data they claim to.

## Scope

The maintainer cut this phase on 2026-11-03. Phase 5 spent most of its time
checking its own checks (four review-fix rounds), so this phase keeps one
simulation check per domain plus the visualization checks.

- **Kept:**
  - planted cells on the real path (encoding);
  - simulated paths with known laps, trials and visits in each declared
    environment (behavior), plus event alignment against numpy;
  - planted misfits that measure how sensitive the environment verdict is;
  - the visualization checks.
- **Moved to the dataset-loaders plan:** the loaders' file truth (trials,
  events, cell labels, file speed, LFP) and the published-result
  reproductions, for the datasets that plan covers.
- **Dropped:**
  - a file-truth pass on the other survey loaders;
  - the messiness inventory and new hazard probes (332 probes exist);
  - the UX run on the documented path (to be done later with the
    `/ux-review` skill);
  - ladder rungs 1 and 3.

## Rules for this phase

- **Every checked operation records `check`:** `agrees`, `disagrees` or
  `no truth`, with the truth's source (`simulation: rung 2`,
  `planted: rung 4`, `naive numpy`, `invariant: split-half > shuffle`). An
  operation without a check is smoke-only and listed as such.
- **Truth comes from outside the code under test:** a simulated or planted
  answer, plain numpy written from the definition (it must not call or copy
  the function under test), or a scientific invariant.
- **Tolerances are written before the run,** next to each check, with the
  reason (bin size, frame interval). A check is never loosened after seeing
  the result, and no parameter is tuned to agree.
- **Every result records the `main` commit it ran on.**
- **Decoding is one metric among several.** It stays, with its check against
  the true position. A write-up never leads with decoding alone, and a
  decoding comparison states its metric and sample set.
- **One review round, one fix round.** After the fix round, only fixes that
  change a committed result go in; the rest are listed as follow-ups.
  Findings about the harness's own yardstick, rather than neurospatial or
  the data, end the review.

## The simulation ladder

| Rung | Behavior | Environment | Tracking | Neurons | What a failure here points at |
| --- | --- | --- | --- | --- | --- |
| 2 | simulated in the **declared real environment** (`simulate_trajectory_ou`, `simulate_trajectory_laps`, `simulate_trajectory_sinusoidal`) | declared | exact | simulated | the library, on the real apparatus geometry |
| 4 | **real** path and time base | declared | real | planted (6b) | the real behavior: coverage, speed, stops, head spread |
| 5 | real | declared | real | **real** | the data, the authors' conventions, or biology |

- **Behavior truth comes from the simulation.** A simulated path has known
  laps, trials, region visits, speed, running bouts and goal arrivals.
- **Seeds.** Every simulated session is seeded, and its parameters are saved
  with the results as the ground truth.

## Tasks

### 6.0 Full rerun on `main`

Rerun every recording on `main` after #281, in both modes (`results/` and
`--inferred-env`). This gives the 11 records that need it the one full run
that `--ops` reruns now require (load fingerprint). Then refresh the
`FINDINGS.md` counts that date from batches 1–5 (recordings, probes, silent
defects). Add `-j` to `run.py` first (recordings in separate processes, 8 at
a time).

### 6b. Encoding: planted cells on the real path (one agent)

For every recording with positions, plant 30 units with known tuning on a
seeded 10-minute slice of the animal's real path, real time base and declared
environment, with `generate_population_spikes(models, times, positions,
headings=...)`. Use the `neurospatial.simulation` cell models that make sense
for the apparatus: place, grid, border, head direction, boundary vector,
object vector. Plant untuned units as negatives. The tracked head position
is the animal's position, as in real work. Save each planted model's
parameters with the results.

| Operation | Check |
| --- | --- |
| `rates.binned_vs_numpy` | `compute_spatial_rates(method="binned")` equals spike-count / occupancy histograms written in numpy, on every recording (float tolerance; NaN where occupancy is 0) |
| `cells.planted` | on the planted units: field centre error and field size against the model; detection rates (planted positives, untuned negatives); grid spacing and orientation; preferred direction; object distance and angle |
| `fields.detect` | `detect_place_fields` field count on planted place cells, and on shuffled spikes (fields must drop) |
| `fields.stability` | split-half rate-map correlation of planted place cells above the circular-shift null |
| `fields.directional` | `compute_directional_place_fields` on linear and W tracks, with planted direction-selective cells: within-direction split-half correlation above cross-direction |
| `rates.tuning_curves` | `compute_tuning_curves` over `path_progress` per trial against per-trial numpy histograms |

### 6c. Behavior, events and environment sensitivity (one agent)

| Operation | Check |
| --- | --- |
| `behavior.simulated` | on rung-2 sessions in each declared environment: `detect_laps`, `detect_trials`, `detect_region_visits`, `compute_speed`, `segment_by_velocity`, `path_progress` and `distance_to_goal` against the simulated laps, trials, visits, speed, bouts and goal arrivals. `tmaze_alternation_session` and `linear_track_session` give whole sessions with known trials |
| `behavior.path_progress` | `path_progress` against the authors' `linearized` coordinate where the loader already has it (11 recordings): rank correlation per trial, and range |
| `events.psth` | `peri_event_histogram` and `align_spikes_to_events` at the recording's events: counts equal numpy counts; the population response at real events is above that at shuffled event times |
| `time.support` | an analysis restricted to run `epochs` uses no spike or sample from sleep or rest, and a result's stated support equals the epochs passed |

**How sensitive the environment verdict is.** On rung-2 sessions in one
apparatus of each kind (square, circle, linear, loop, W, radial or star,
labyrinth, lattice, 3-D room), plant misfits of known size: a shift of
2–10 cm, a scale of ±5–30 %, a rotation of 3–10° about the centre, a wrong
width, and a missing or extra arm. Run the declared-environment checks
(`judge_declared`, the misfit refusal, the frame offset) with tracking spread
added as the real recording of that kind measured it. Report the smallest
misfit each check catches. This settles two limits set after seeing real
data: the 20-bin judged stretch (it was 10) and the bias limit in bins.

### 6f. Visualization: does the figure show the data (one agent)

Public surface: the `plot` / `plot_field` / `animate_fields` methods on the
environment and result classes, the free plotting functions
(`plot_head_direction_tuning`, `plot_object_vector_tuning`,
`plot_phase_precession`, `plot_population_coverage`,
`plot_peri_event_histogram`, `plot_peri_event_response`, `plot_regions`,
`plot_circular_basis_tuning`), and `neurospatial.animation`.

**Static figures.** Check on rung-2 sessions first, where every field is
known, then on every recording. Render headless with `MPLBACKEND=Agg` and
read the artists back:

| Check | How |
| --- | --- |
| The image is the data | the plotted array (`AxesImage.get_array()`, mesh or collection values) equals the result's values bin for bin |
| Missing is not zero | NaN bins are masked or drawn in a distinct "no data" colour, never the colour of 0 Hz |
| Orientation and extent | the axes limits contain the environment's extent in its units; 2-D is equal-aspect; a sample at max y is drawn at the top; pixel-unit recordings say so |
| Peak in the right place | the brightest bin's position in data coordinates equals the rate argmax position |
| Angles | a polar or heading plot puts 0 at East (allocentric) or ahead (egocentric); a unit with a known preferred direction peaks there |
| Units and labels | axis and colorbar labels carry the physical units |
| Track layouts | `plot_linear_layout` order and gaps match the linearization used |
| 3-D | the bat room and the lattice render with the declared extent |

**Visual review.** For each recording, write a contact sheet (one PNG grid of
its figures) to `results/figures/`. The agent reads each sheet and writes what
a scientist would see wrong. It sends the maintainer a short set ordered by
risk: each figure type once, every sheet where an automated check failed, and
the 3-D, pixel-unit and track recordings.

**Animation.** Export a 10 s slice per apparatus kind (9) with
`Animation.export` and the `widget` backend, with a position overlay, a spike
raster and events where present. Check: the frame count equals duration ×
fps; at sampled frames the position marker's pixel maps back to the recorded
position within one bin; spikes and events appear on the frame for their
time. Try `napari` and `fastplotlib` off-screen for at most 30 min, checking
a captured frame is not blank; otherwise mark them "ran only".

### 6g. Findings

A FINDINGS section per domain: what agrees with the truth, what disagrees
(each a gap with evidence), and what stays smoke-only. The matrix gets a
check-coverage column by domain.

## Budget

**Measured.** The driver takes about 60 min serially on 37 recordings
(`replay.compare_methods` and `rates.egocentric_boundary` the largest). Shuffle
verdicts cost about 2–3 ms per unit per shuffle. Peak memory is at most
3.7 GB per recording; the machine has 18 cores and 64 GB.

**Limits.**

| Item | Limit |
| --- | --- |
| Shuffle verdicts | at most 50 units per recording (seeded sample), 200 shuffles, declared environment only |
| Planted units | 30 per recording, on a seeded 10-min slice |
| Rung 2 sessions | one per apparatus kind (9), 10 min each |
| Replay | smoke only: 20 candidate events, 50 shuffles |
| Animation exports | one 10 s slice per apparatus kind |
| Interactive viewers | 30 min, then "ran only" |

**Targets.**

- 6.0: under 1 h wall time with `-j 8`.
- 6b, 6c and 6f: three agents in parallel, 2–3 h each, timeboxed at 3 h.
  When the timebox runs out, the agent stops and reports what is done.
- One review round and one fix round for the whole branch: about 1 h.
- 6g: about 1 h.

Only one agent runs the driver at a time (overlapping runs starve the CPU).

## How it ships

One PR for the whole phase, from one integration branch:

1. 6.0 on the integration branch first.
2. Split `datasets/operations.py` by domain (`operations_encoding.py`,
   `operations_behavior.py`, `operations_figures.py`), keeping the shared
   helpers and the registry where they are, and assign gap ids per domain.
3. 6b, 6c and 6f, each in its own worktree branched from the integration
   branch, merged back when done.
4. One independent review of the combined branch, one fix round, then 6g.

`results/matrix.md`, `warnings.md` and `verdicts.json` are rendered once at
merge (`run.py --render-only`). Each domain brief lists its required checks
in priority order, so an agent cannot fill its time with whatever is easiest
to score.

## Risks

- **Self-consistency checks.** These happened twice in phase 5. The review
  asks of every `agrees`: does this truth depend on the code it checks?
- **Head tracking.** The tracked point's offset and spread widen fields and
  add speed noise at rest. Planted cells on the same path are the comparison
  before a real-data difference is called a library bug.
- **Headless rendering on macOS:** napari can return black frames,
  fastplotlib needs a GPU context, ffmpeg must be present, and reading
  figures back depends on the matplotlib version (record it with the
  results).
- **The API changes during the phase.** The `main` commit is recorded per
  result. A library bug found mid-phase is a gap; a harness workaround for it
  is listed, never a silent fix.

## Deliberately not in this phase

- **Source changes.** Each disagreement becomes a gap and goes to the source
  plan.
- **New recordings and new decoding methods.**
- **Comparing replay methods.** The methods port published implementations,
  so `replay.compare_methods` stays a smoke run. Seen while sizing this
  phase: on 40 events of `grosmark_000044_linear`, the `davidson_2009`
  unit-identity null gave a mean z of −0.66, so no event passed all three
  nulls. That belongs to the replay source work.
- **What the scope cut dropped** (see [Scope](#scope)).

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_checks.py::test_binned_rates_match_numpy` | the numpy reference and `compute_spatial_rates(method="binned")` agree on a simulated recording, and the check reports `disagrees` when the reference is perturbed |
| `test_checks.py::test_check_without_truth_is_no_truth` | an operation with no truth records `check="no truth"`, never `agrees` |
| `test_checks.py::test_naive_reference_is_independent` | the numpy references import nothing from `neurospatial` beyond the inputs they are given |
| `test_checks.py::test_planted_place_cell_recovered` | a place cell planted on a simulated path is found with its centre within one bin, and an untuned planted unit is not |
| `test_checks.py::test_simulated_laps_counted` | `detect_laps` on a `simulate_trajectory_laps` path in a declared loop counts the simulated laps |
| `test_checks.py::test_psth_counts_match_numpy` | PSTH counts equal numpy counts; shuffled events give a flat population response |
| `test_checks.py::test_planted_shift_caught` | a declared square shifted by more than the frame-offset limit on a rung-2 session is not judged `matches` |
| `datasets/tests/test_figures.py::test_plotted_array_is_the_data` | the rate-map figure's image array equals the rates and NaN bins are masked; a flipped y axis fails the orientation check |
| `test_figures.py::test_export_frames_follow_position` | a short simulated export has duration × fps frames and the position marker maps back to the recorded position within one bin |
| ruff check and format on `datasets` | clean |
| Real run (manual) | every checked operation has `check` set; the matrix shows check coverage by domain |

## Review

One independent reviewer against the combined diff, given the code, not the
agents' conclusions. Confirm:

- every `agrees` compares against a truth that does not come from the code
  under test;
- decoding is never the only evidence in a write-up, and every decoding
  comparison is like with like;
- the numbers in the write-ups match the results JSON.
