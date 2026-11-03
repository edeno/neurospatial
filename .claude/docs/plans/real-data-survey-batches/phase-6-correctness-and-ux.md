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
- **Truth comes from outside the code under test,** in this order: the file
  (the authors' linearization, trials, labels, events, speed); the paper (a
  number read in the text, cited under the cite-only-verified rule); an
  independent naive computation (plain numpy written in the check, not
  calling the function under test); a scientific invariant (a shuffle or
  split-half baseline).
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

A loader that cannot find one says so in its docstring. No guessed values.

### 6b. Encoding checks (one agent)

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

### 6c. Behavior and events checks (one agent)

| Operation | Check |
| --- | --- |
| `behavior.speed` | `compute_speed` against `file_speed`, and against a finite difference on the same smoothing |
| `behavior.path_progress` | `path_progress` against the authors' `linearized` (rank correlation per trial, and range) |
| `behavior.trials` | `detect_trials` and `detect_region_visits` against the `trials` table (match rate, start/stop error in s) |
| `behavior.laps` | `detect_laps` (with `start_region=`) against path truth (keep the existing study) |
| `behavior.running` | `segment_by_velocity` running bouts against the file's run epochs where stated |
| `behavior.goal` | `distance_to_goal`, `time_to_goal` on the water maze against its platform and the file's escape latency |
| `events.psth` | `peri_event_histogram` and `align_spikes_to_events` at `events`: counts against numpy counts; the population response at real events above shuffled event times |
| `regions.visits` | `regions_containing_points` / region entries at `regions` against `events` (well entries) |
| `behavior.decision` | VTE and pre-decision metrics at W-track and bandit choice points: smoke-only unless a file truth is found (say so) |

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
clipped fields, misleading colour scales, empty panels. The sheets are
published to the maintainer for an eye check. Automated checks don't replace
this.

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
| `test_calibration.py::test_incidental_catch` | a raise whose message does not name the hazard records `caught (incidental)` |
| ruff check and format on `datasets` | clean |
| Real run (manual) | every operation has `check` set; the matrix shows check coverage by domain |

## Review

Before opening each PR, dispatch an independent reviewer against the diff.
Give it the code, not the agent's conclusions. Confirm:

- every `agrees` compares against a truth that does not come from the code
  under test (no self-consistency checks);
- every paper number was read (cite only verified);
- decoding is never the only evidence in a write-up, and every decoding
  comparison is like with like;
- the numbers in the write-ups match the results JSON.
