# Phase 6 — Does it give the right answer: encoding, behavior, events, visualization, messiness, UX

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [appendix](appendix.md)

Prerequisite: phase 5 merged, including the tracked-point pass (every loader
records what was tracked; one declared-environment verdict that separates
apparatus misfit from tracking spread).

## Why this phase

Today an operation is `ok` when the call does not raise. Only decoding has a
truth built in (the true position), so decoding error became the number every
batch compares. The survey calls 12 of the 175 public functions in
`encoding`, `behavior`, `events`, `regions` and `stats`: no `events`
function, no `regions` function, 3 of about 60 `behavior` functions, and no
place-field detection, stability, directional fields, phase precession or
object-vector scores. Most files carry truth the loaders drop: authors'
linearization (11 recordings), trial tables, epochs, cell-type and module
labels, the file's own speed, reward and well events, LFP.

Goal: every operation the survey runs is checked against something that does
not come from the code under test, the survey covers the encoding and
behavior workflows a spatial neuroscientist runs, the figures and animations
show the data they claim to, real-data messiness is
measured and probed, and the documented path is tried as written.

## Rules for this phase

- **Every operation has a check.** A result records `status` (ran / error /
  n/a) and `check`: `agrees`, `disagrees` or `no truth`, with the truth's
  source (`file: trials table`, `paper: Mizuseki 2012 Table 1`, `naive numpy`,
  `invariant: split-half > shuffle`). An operation added without a check is
  smoke-only and is listed as such in the matrix.
- **Truth comes from outside the code under test.** In order of preference:
  - **the file:** the authors' linearization, trials, labels, events, speed,
    ripples;
  - **simulation:** a rung of the simulation ladder below, from simulated
    neurons on the real path up to a fully simulated session in the declared
    real environment;
  - **the paper:** a number read in the text, cited under the
    cite-only-verified rule;
  - **an independent naive computation:** plain numpy written from the
    definition in the paper or the docstring. It must not call the function
    under test or copy its code, because a copy repeats the same mistake;
  - **a scientific invariant:** a shuffle or split-half baseline.
- **Tolerances are written before the run,** next to each check, with the
  reason (bin size, frame interval, the authors' stated criterion). A check
  is never loosened after seeing the result. When the authors used a
  different definition (speed threshold, bin size, cell criterion), the check
  uses theirs where it is stated, and otherwise reports the difference
  instead of `disagrees`.
- **No tuning to agree.** A parameter changed so that a number matches is a
  finding, not a fix.
- **Every result records the `main` commit it ran on,** so a result can be
  traced while the API changes under the survey.
- **Decoding is one metric among several.** Decoding error is a good summary
  of how much spatial information a population carries across many brain
  areas. It stays, with its check against the true position. It must not be
  the only evidence. A write-up that reports what changed reports the
  encoding and behavior checks alongside it, and never leads with decoding
  alone. A decoding comparison states its metric (2-D or along the track) and
  its sample set, so that two numbers are only compared when both are the
  same.
- **Agents work by domain, not by dataset.** Each domain agent covers every
  recording that carries the truth its checks need.

## The simulation ladder

Each rung adds one layer of reality to a session whose answer is known. When
a check passes on one rung and fails on the next, the layer just added is the
cause.

| Rung | Behavior | Environment | Tracking | Neurons | What a failure here points at |
| --- | --- | --- | --- | --- | --- |
| 1 | simulated | simulated apparatus (`simulation.mazes`, arenas) | exact | simulated | the library |
| 2 | simulated in the **declared real environment** (`simulate_trajectory_ou`, `simulate_trajectory_laps`, `simulate_trajectory_sinusoidal`) | declared | exact | simulated | the apparatus geometry and how the environment captures it |
| 3 | as rung 2 | declared **and** built from the simulated samples | **messy**: a head point offset from the body in the heading direction and jittered sideways, gaps, LED jumps, timestamp jitter, pixel units | simulated | real-data messiness and the samples-built environment |
| 4 | **real** path and time base | declared and samples-built | real | simulated (planted, 6b) | the real behavior: coverage, speed, stops, head spread |
| 5 | real | declared and samples-built | real | **real** | the data, the authors' conventions, or biology |

- **Calibrate the messy rung to the real data.** Rung 3 takes its messiness
  from what the inventory (6d) measures on each recording: head-point offset
  and spread split by speed, gap lengths, jump rates and frame-interval
  jitter. Rung 3 is then that recording's messiness on a known answer.
- **Behavior truth comes from the simulation.** A simulated path has known
  laps, trials, region visits, speed, running bouts and goal arrivals. So
  `detect_laps`, `detect_trials`, `detect_region_visits`, `compute_speed`,
  `segment_by_velocity`, `path_progress` and `distance_to_goal` are checked
  on rungs 2–3 in every declared environment, including the ones whose file
  has no trials table. `tmaze_alternation_session` and `linear_track_session`
  give whole sessions with known trials.
- **Environment capture is scored on rungs 2–3.** Build the samples-built
  environment from simulated samples in a known apparatus, then measure:
  - outline coverage;
  - shortcuts through walls or between bars, as connectivity against the
    true graph;
  - the analyses on top (planted fields, laps, decoding) against the truth,
    on the declared environment and the samples-built one.

  The real-data fidelity study answers the same question without a known
  answer.
- **Replay** is planted on rungs 4–5 with `simulate_replay_event` in quiet
  periods (6c).
- **Figures** (6f) are checked on rung 1–2 sessions first, where every field
  is known, before they are checked on real data.
- **Seeds.** Every simulated session is seeded, and its parameters are
  saved with the results as the ground truth.

## Tasks

### 6a. Loaders hand on the file's truth (one agent per loader group)

Each loader puts what its file already holds into `Recording.params` under
shared names, and declares it in `EXPECTATIONS` first:

| Name | Content | Known to exist in |
| --- | --- | --- |
| `linearized` | authors' linear position | 11 recordings (already) |
| `trials` | start/stop (and outcome, arm, well) per trial | 000447, figure-8, Agarwal, water maze, J16, 001880 |
| `events` | named event times: reward, well entry, port poke, light, object | J16 ports, 000447 wells, 001880 reward, water maze platform |
| `regions` | stated wells, ports, reward zones, start box, platform, as polygons in position units | same as `events` |
| `cell_labels` | authors' unit labels: cell type, module, region, grid/HD flags | Gardner, hc-11, hc-3, Vollan, Peyrache, Alexander EBC |
| `file_speed` | the file's speed | J16, 000447, Vollan |
| `lfp` | one theta-bearing channel and its sampling rate (CA1 pyramidal layer when stated), streamed slice only | hc-3, 000044, Vollan, J16 where present |
| `published` | numbers read in the paper, each with its citation: place-cell fraction, field size, HD-cell count, grid spacing | per paper |
| `ripples` | the authors' ripple or candidate-event times | the 4 recordings with `has_ripples` (already partly) |
| `epochs` | named run, sleep and rest epochs with their times | figure-8, hc-18, hc-11, Peyrache, Shipley (already partly) |

A loader that cannot find one says so in its docstring. No guessed values.

### 6b. Encoding checks (one agent)

**Planted cells on real behavior.** For every recording, plant units with
known tuning on the animal's real path, real time base and declared
environment, with `generate_population_spikes(models, times, positions,
headings=...)`. Use the cell models in `neurospatial.simulation`: place,
grid, border, head direction, boundary vector, object vector and spatial
view. Plant only the cell types that make sense for the apparatus.

- **Seeds and ground truth.** Planted spikes use a fixed seed. Each planted
  unit's model parameters are saved with the results as the ground truth.
- **Every encoding operation runs on the planted units,** alongside the
  recorded ones:
  - field centre error and field size against the model;
  - detection rates (planted positives, and planted untuned units as
    negatives);
  - grid spacing and orientation;
  - preferred direction;
  - object distance and angle;
  - phase precession, where a theta phase is planted with `add_modulation`.
- **Two jobs.** This is the truth for operations no file can check. It is
  also the fair comparison of declared and samples-built environments: the
  same planted cells, scored on both.
- **Head tracking.** Planted cells use the tracked head position as the
  animal's position. That is exactly what a real analysis does, so the
  tracking spread found in phase 5 shows up here as it would in real work.

| Operation | Check |
| --- | --- |
| `rates.binned_vs_numpy` | `compute_spatial_rates(method="binned")` equals spike-count / occupancy histograms written in numpy, on every recording (exact up to float tolerance; NaN where occupancy is 0) |
| `fields.detect` | `detect_place_fields` and `field_size`: field count and size per cell against `published`, and on shuffled spikes (fields must drop) |
| `fields.stability` | `field_stability` / split-half rate-map correlation: place cells above the circular-shift null |
| `fields.directional` | `compute_directional_place_fields` on linear and W tracks: within-direction split-half correlation above cross-direction correlation (unidirectional fields on tracks) |
| `cells.labels` | `is_place_cell`, `is_head_direction_cell`, `grid_score`, `border_score` against `cell_labels` (confusion table; interneurons as the negative class where labelled) |
| `cells.population` | `population_vector_correlation` across sessions: novel vs familiar W-track (000447), split-halves of one session above across-environment |
| `theta.phase_precession` | `compute_phase_precession` on track place cells with `lfp`: share of negative slopes against a spike-time shuffle |
| `cells.object_vector` | `object_vector_score` / `is_object_vector_cell` on Kanter with its objects |
| `rates.tuning_curves` | `compute_tuning_curves` over `path_progress` per trial against per-trial numpy histograms |
| `cells.planted` | every encoding operation above on the planted units, against the planted models (declared and samples-built environments) |

**One published result per paper.** For each paper with a readable
Methods section, reproduce one result end to end with the paper's stated
parameters. Candidates:

- Gardner 2022 module spacings;
- the place-cell fraction in Mizuseki 2012 or Grosmark 2016;
- the HD tuning width in Peyrache 2015;
- the border-cell share in Solstad-style data.

Report our number, theirs, and every parameter we could not match. This is
the end-to-end test a scientist would use to judge the package.

### 6c. Behavior and events checks (one agent)

| Operation | Check |
| --- | --- |
| `behavior.simulated` | every behavior operation below on rung 2–3 sessions in each declared environment, against the simulated laps, trials, visits, speed, bouts and goal arrivals |
| `behavior.speed` | `compute_speed` against `file_speed`, and against a finite difference on the same smoothing |
| `behavior.path_progress` | `path_progress` against the authors' `linearized` (rank correlation per trial, and range) |
| `behavior.trials` | `detect_trials` and `detect_region_visits` against the `trials` table (match rate, start/stop error in s) |
| `behavior.laps` | `detect_laps` (with `start_region=`) against path truth (keep the existing study) |
| `behavior.running` | `segment_by_velocity` running bouts against the file's run epochs where stated |
| `behavior.goal` | `distance_to_goal`, `time_to_goal` on the water maze against its platform and the file's escape latency |
| `events.psth` | `peri_event_histogram` and `align_spikes_to_events` at `events`: counts against numpy counts; the population response at real events above shuffled event times |
| `regions.visits` | `regions_containing_points` / region entries at `regions` against `events` (well entries) |
| `behavior.decision` | VTE and pre-decision metrics at W-track and bandit choice points: smoke-only unless a file truth is found (say so) |
| `replay.candidates` | candidate events against the authors' `ripples` (match rate, onset error); replay detection on events planted with `simulate_replay_event` in the real recording's quiet periods (detection rate and false positives on unplanted periods). Stationary decoded events are not false positives |
| `time.support` | results keep the time support of their inputs: an analysis restricted to run `epochs` uses no spike or sample from sleep or rest, and a result's stated support equals the epochs passed |

**Round trips.** Extend `io.nwb_roundtrip` beyond spikes and positions:
declared environments, regions, trials and events are written to NWB, read
back, and compared field by field. Do the same through pynapple where the
reader supports it.

### 6d. Real-data messiness (one agent)

- **Inventory.** For each recording, measure the messiness it actually has:
  - tracking gaps (count, longest);
  - jumps beyond a physical speed (LED reflections, swaps);
  - duplicated or non-monotonic timestamps;
  - irregular frame intervals;
  - spikes outside the tracking span;
  - concatenated epochs with gaps;
  - units with no spikes in the epoch;
  - head-point spread (from phase 5).

  Write it as a table in `results/studies/messiness.md`.
- **Probes.** Extend the hazards (today: degrees, missing-as-zero, spikes in
  ms, length ×100) with the messiness found:
  - tracking jumps;
  - duplicated timestamps;
  - an epoch gap;
  - spikes beyond tracking;
  - irregular sampling.

  Each gets a verdict per affected operation: caught (a warning or error that
  names the cause), silent, or no effect.
- **No caught by accident.** A raise whose message does not name the cause is
  `caught (incidental)`, not `caught`.
- **Reproducibility.**
  - Every seeded operation is run twice, and the outputs must be identical.
  - Each result is checked for what it records about how it was made: input
    selections, exclusions, parameters, seed and versions. Anything a result
    needs to be reproduced but does not carry is a finding.
- **Cost.** Record runtime and peak memory for every operation, as the
  driver already does, on the largest recordings: Vollan (about 600 units),
  J16 (500 Hz), the 1000-shuffle cell verdicts on every unit, and the
  animation exports. Time each new domain on one recording before its full
  run, and extrapolate before launching.

### 6e. UX: the documented path on real data (one agent)

- **Run the documented path.** For each recording, run the getting-started
  path in `docs/user-guide/`, as written, replacing only the data. Record
  every change the agent had to make (a workaround, an extra argument, a
  conversion) as a UX finding with the doc and line.
- **Rate every message.** Collect every warning and error the survey
  produced across all operations. Rate each on what failed, why, and how to
  fix it (`.claude/rules/public-api.md` contract), and list the ones that fail
  any of the three.
- **Count the call sites.** Count the lines each operation in
  `operations.py` needs beyond the public call (glue, conversions, private
  imports). Anything above a few lines is a UX finding.

### 6f. Visualization: does the figure show the data (one agent)

The survey calls no plotting or animation code today. Public surface:

- 35 `plot` / `plot_field` / `animate_fields` methods on the environment and
  result classes (rates, directional, egocentric, boundary-vector, view,
  tuning, comparison, decoding, events, replay, shuffle, layout
  `plot_linear_layout`);
- the free plotting functions (`plot_head_direction_tuning`,
  `plot_object_vector_tuning`, `plot_phase_precession`,
  `plot_population_coverage`, `plot_peri_event_histogram`,
  `plot_peri_event_response`, `plot_regions`, `plot_circular_basis_tuning`);
- `neurospatial.animation` (`Animation`, its overlays and `export`).

**Static figures.** For every recording, every plot that applies is
rendered headless with `MPLBACKEND=Agg`. Each figure is checked by reading
back the artists, not by trusting the call:

| Check | How |
| --- | --- |
| The image is the data | the plotted array (`AxesImage.get_array()`, mesh or collection values) equals the result's values bin for bin |
| Missing is not zero | NaN bins are masked or drawn in a distinct "no data" colour, never the colour of 0 Hz |
| Orientation and extent | the axes limits contain the environment's extent in its units; 2-D is equal-aspect; a sample at the top of the arena (max y) is drawn at the top; pixel-unit recordings say so |
| Peak in the right place | the brightest bin's position in data coordinates equals the rate argmax position |
| Angles | a polar or heading plot puts 0 at East (allocentric) or ahead (egocentric) per the conventions; a unit with a known preferred direction peaks there |
| Units and labels | axis and colorbar labels carry the physical units (cm, Hz, s, rad) |
| Track layouts | `plot_linear_layout` order and gaps match the linearization used |
| 3-D | the bat room and the lattice render with the declared extent |

**Visual review.** For each recording, a contact sheet (one PNG grid of its
figures) is written to `results/figures/`. The agent reads each sheet as an
image and writes what a scientist would see wrong: unreadable defaults,
clipped fields, misleading colour scales, empty panels.

The maintainer's time is the limit, so the agent sends a short set ordered
by risk:

- each figure type once;
- every sheet where an automated check failed;
- the 3-D, pixel-unit and track recordings.

The rest stay in `results/figures/` to look at on request. An agent reading
an image misses subtle problems, so the automated checks and the
maintainer's look are both needed.

**Animation.**

- **Export a short slice.** For each recording, export a 10 s slice with
  `Animation.export` (`.mp4` and `.gif`) and the `widget` backend, with a
  position overlay, a spike raster and events where the recording has them.
- **Check the frames against the data:**
  - the frame count equals duration × fps;
  - at sampled frames, the position marker's pixel maps back to the recorded
    position at that frame's time (within one bin);
  - spikes and events appear on the frame for their time.
- **Interactive viewers.** `napari` and `fastplotlib` are tried off-screen
  where they can render headless. Where they can't, they are smoke-only and
  marked so. Earlier work found that napari can return black screenshots off
  screen, so check a captured frame is not blank before trusting it.

**UX.** For each figure type, count the lines needed from result to readable
figure, and log the defaults a scientist would have to change (titles, units,
colour range, NaN colour).

### 6g. Findings

- **A FINDINGS section per domain:** what agrees with the truth, what
  disagrees (each a gap with evidence), and what stays smoke-only.
- **Matrix coverage:** a column for check coverage: share of operations with
  a check, by domain.

## Budget

The phase must not take much longer than the earlier ones. The limits below
come from measurements taken when this was written.

**Measured.**

- **The driver today:** about 60 min of compute for the operations on 37
  recordings, serially, before probes and data loading. The largest costs are
  `replay.compare_methods` (27 min, mostly 000044 circular and linear) and
  `rates.egocentric_boundary` (8 min).
- **Shuffle cell verdicts:** about 2–3 ms per unit per shuffle. Vollan's 644
  units at the default 1000 shuffles take about 29 min, for one cell type on
  one environment.
- **Peak memory:** at most 3.7 GB per recording (Gardner); the median is
  0.13 GB. The machine has 18 cores and 64 GB.
- **Agent time in phase 5:** 1.7–2.3 h of wall time per agent, then 20–40
  min per review-fix round.

**Limits.**

| Item | Limit | Why |
| --- | --- | --- |
| Shuffle verdicts in the survey | at most 50 units per recording (seeded sample), 200 shuffles, declared environment only | ≈ 27 s per cell type per recording instead of up to 29 min. The full 1000 runs only in the published-result reproductions |
| Planted units | 30 per recording, on a seeded 10-min slice of the real path | enough to score detection and centre error |
| Ladder rungs 1–3 | one recording per apparatus kind: square, circle, linear, loop, W, radial or star, labyrinth, lattice, 3-D room (9). Rung 4 on every recording | the rungs test kinds of apparatus, not each file |
| Simulated sessions | 10 min each | |
| `--inferred-env` run | only operations that take an environment | the I/O and reader operations don't change |
| Hazard probes | not on shuffle verdicts, replay or animation | these are the slow operations, and the probes ask about warnings, not results |
| Replay | at most 200 candidate events per recording | about a quarter of today's compute |
| LFP | one channel, one 10-min streamed slice | |
| Animation exports | one 10 s slice per apparatus kind (9) | |
| Published-result reproductions | 3 papers, one result each | open-ended otherwise |
| Interactive viewers | 30 min to get off-screen rendering working, then "ran only" | headless GPU is a known time sink |

**Parallel driver.** `run.py` gets a `-j` option that runs recordings in
separate processes, 8 at a time. That fits even if all 8 peaked like Gardner.
Domain agents share the machine, so only one agent runs the driver at a time
(earlier work saw overlapping runs starve the CPU).

**Targets.**

- **A full driver rerun** of every operation, the probes and ladder rungs 1–4
  on all recordings: under 1 h wall time with `-j 8`. Before launching, each
  domain times one recording, and the agent extrapolates. If the
  extrapolation exceeds the target, the limits above are tightened and the
  change is stated.
- **Agent time:**
  - 6a takes 2–3 h, with three agents over loader groups;
  - 6b–6f take 2–3 h each in parallel, plus one review-fix round;
  - 6g takes about 1 h.

  That is about one long working day of wall time.
- **Timeboxes.** Each domain agent gets a 3 h timebox and its required
  checks in priority order. When the timebox runs out, it stops and reports
  what is done and what is left; it does not keep going.

**The likely overruns:**

- 6a: 37 loaders re-read for their truth, with CRCNS and DANDI fetches;
- the published reproductions;
- the headless viewers.

The limits above cap each of these.

## How it ships

One PR for the whole phase, from one integration branch. The tasks run in
this order:

1. 6a (loaders), which every later task depends on, merged into the
   integration branch first.
2. 6b, 6c, 6d, 6e and 6f, each in its own worktree branched from the
   integration branch, and merged back into it when its review is done.
3. 6g (findings), written on the integration branch.

Each task gets its independent review before it merges into the
integration branch, and the whole branch gets one more before the PR.

Before the parallel tasks start:

- **Split the operations by domain.** `datasets/operations.py` keeps the
  shared helpers and the registry; each domain adds its operations in its
  own module (`operations_encoding.py`, `operations_behavior.py`,
  `operations_figures.py`, …).
- **Assign gap ids per domain.**
- **Render once.** `results/matrix.md`, `warnings.md` and `verdicts.json`
  are rendered once, at merge (`run.py --render-only`), never edited by hand.

Each domain brief lists its required checks, so an agent cannot fill its
time with whatever is easiest to score.

## Risks

- **The authors' truth uses other definitions.** Speed thresholds, bin
  sizes, cell criteria and session pooling differ from ours, so "disagrees"
  often means "defined differently". Mitigation: pre-registered tolerances,
  the authors' parameters where stated, and distributions instead of
  pass/fail where the definitions differ.
- **Self-consistency checks.** This happened twice in phase 5 (arena
  outlines, track references). Mitigation: the review question "does this
  truth depend on the code it checks?" for every `agrees`.
- **Head tracking.** The tracked point's offset and spread widen fields,
  shift direction-specific fields and add speed noise at rest. Mitigation:
  report checks split by speed where it matters, and compare against planted
  cells on the same path before calling a real-data difference a library
  bug.
- **Access.**
  - LFP is large, so stream slices only; DANDI may throttle.
  - CRCNS needs its login.
  - Several papers could not be read (Current Biology for 000987, Neuron
    2026 for 001880, the Drieu 2018 supplement). Their numbers stay out
    under the cite-only-verified rule.
- **Headless rendering on macOS.**
  - napari can return black frames.
  - fastplotlib needs a GPU context.
  - ffmpeg must be present.
  - Reading figures back depends on the matplotlib version.

  Mitigation: blank-frame checks, "ran only" marking where a viewer can't
  render, and recording the matplotlib version with the results.
- **The API changes during the phase** (two PRs landed the day this was
  written). Mitigation: record the `main` commit per result. A library bug
  found mid-phase is logged as a gap; a harness workaround for it is a UX
  finding, not a silent fix.
- **Compute.** Many more operations × about 37 recordings × two environment
  kinds × probes × shuffles. Mitigation: the limits and the parallel driver
  in [Budget](#budget), and the one-recording timing before each full run.
- **Triage time.** Deciding whether the library, the data or the authors'
  convention is at fault takes the maintainer's time. Mitigation: every
  `disagrees` comes with its evidence, its likely cause, and what was ruled
  out.

## Deliberately not in this phase

- **Source changes.** Each disagreement becomes a gap and goes to the source
  plan.
- **New recordings,** unless a domain needs a truth no current recording has
  (for example object-vector labels, theta LFP). Then add the one recording
  and say why.
- **New decoding methods.** The existing `decode.position` check stays and
  is reported alongside the other checks.

## Validation slice

| Test | Asserts |
| --- | --- |
| `datasets/tests/test_checks.py::test_binned_rates_match_numpy` | the numpy reference and `compute_spatial_rates(method="binned")` agree on a simulated recording, and the check reports `disagrees` when the reference is perturbed |
| `test_checks.py::test_check_without_truth_is_no_truth` | an operation with no truth available records `check="no truth"`, never `agrees` |
| `test_checks.py::test_trials_match_rate` | `behavior.trials` on a simulated recording with a known trials table reports full agreement, and a shifted table reports start error equal to the shift |
| `test_checks.py::test_psth_counts_match_numpy` | PSTH counts equal numpy counts; shuffled events give a flat population response |
| `datasets/tests/test_figures.py::test_plotted_array_is_the_data` | on a simulated recording, the rate-map figure's image array equals the rates and NaN bins are masked; a figure with a flipped y axis fails the orientation check |
| `test_figures.py::test_export_frames_follow_position` | a short simulated export has duration × fps frames and the position marker maps back to the recorded position within one bin |
| `test_checks.py::test_ladder_isolates_messiness` | a simulated session in a declared square passes the place-field check on rung 2 and reports the injected head offset's effect on rung 3, with the same seed |
| `test_checks.py::test_simulated_laps_counted` | `detect_laps` on a `simulate_trajectory_laps` path in a declared loop counts the simulated laps |
| `test_checks.py::test_planted_place_cell_recovered` | a place cell planted on a simulated path is found with its centre within one bin, and an untuned planted unit is not |
| `test_checks.py::test_naive_reference_is_independent` | the numpy references in the checks import nothing from `neurospatial` beyond the inputs they are given |
| `test_checks.py::test_seeded_rerun_is_identical` | a seeded operation run twice gives identical outputs |
| `test_checks.py::test_nwb_roundtrip_regions_trials` | regions, trials and events survive a write and read unchanged |
| `test_calibration.py::test_incidental_catch` | a raise whose message does not name the hazard records `caught (incidental)` |
| ruff check and format on `datasets` | clean |
| Real run (manual) | every operation has `check` set; the matrix shows check coverage by domain |

## Review

Before each task merges into the integration branch, and once more before
the PR, dispatch an independent reviewer against the diff. Give it the code,
not the agent's conclusions. Confirm:

- every `agrees` compares against a truth that does not come from the code
  under test (no self-consistency checks);
- every paper number was read (cite only verified);
- decoding is never the only evidence in a write-up, and every decoding
  comparison is like with like;
- the numbers in the write-ups match the results JSON.
