# Real-data survey: findings

Eight public recordings were loaded into plain arrays and run through 22
neurospatial operations, each recording was re-run with known defects put back
(58 probes), and three results were compared with pynapple and opexebo. This
document ranks what that showed into proposed changes to neurospatial. The
evidence lives in [GAPS.md](GAPS.md) (gap entries `G-###`) and
[results/](results/) (`matrix.md`, `warnings.md`, `reference_checks.md`, one
JSON per dataset); every number below can be traced there.

Datasets: DANDI 000582 (rat MEC open field), OSF Mouse32 (mouse ADn head
direction), DANDI 000552 (mouse CA1 figure-8, ripples, sleep), the Rosenberg
2021 labyrinth (behavior only, binary-tree maze), the Yartsev 2025 bat demo
(3-D flight), DANDI 002030 (mouse CA1 miniscope ΔF/F), DANDI 000405 (rat open
field, binned rates, mislabelled tracking) and DANDI 001701 (mouse MEC/CA1 on an
X maze, mislabelled head direction).

## At a glance

- **The common path holds up.** Environments, occupancy, rate maps, spatial
  information, decoding, NWB round trips and behavior segmentation ran on every
  recording that had the inputs, in 1-D, 2-D and 3-D, in seconds and under
  300 MB (traced). Decoding is accurate where the data allow it (figure-8
  8.0 cm, 001701 3.5 cm median error).
- **Warnings are not too aggressive.** Across the eight clean recordings only
  one neurospatial warning fired that should not have (the spike-drop warning
  when tracking covers part of the session, G-001, being fixed).
- **Warnings are not aggressive enough.** Of 58 probes, 19 defects passed
  silently and changed the result. Four simple checks, calibrated below,
  would catch most of them without firing on any clean recording.
- **Two wrong results need fixing first**: `read_head_direction` turns radians
  labelled "degrees" into a 6° arc, after which every unit looks like a sharp
  head-direction cell (G-030/G-031); and `detect_laps` finds laps where there is
  no loop (G-015, being fixed).
- **Two candidate wrong results need a decision**: grid scores differ from
  opexebo by up to 0.38 on the same map (G-006), and spatial information on
  sparse 3-D occupancy is at the shuffle level while its bias estimate is small
  (G-022).
- **Biggest missing capability**: spatial maps from calcium imaging or binned
  rates (G-029); every rate function takes spike times.

Each proposal below states the problem, the evidence, the change, the module,
and whether results change for existing users. Proposals are ordered by
scientific risk, then by how many datasets showed the problem. Where one
dataset is the evidence, the section says why that is enough.

## Wrong results

### 1. Refuse a "degrees" label on head direction that is radians (G-030, G-031)

- **Problem.** `read_head_direction` trusts a series' unit label, so radians
  labelled "degrees" come back as a 6° arc, and downstream nothing notices.
- **Evidence.** 001701: values −3.140 to 3.142 labelled degrees; the reader
  returns −0.055 to 0.055 rad. `compute_directional_rates` on those headings
  gives a median mean vector length of 0.999 over 320 units (true 0.12): every
  unit looks like a sharp head-direction cell. One dataset is enough: the
  conversion is deterministic, so every file with this mislabel is wrong in the
  same way (001701 has 218 files; only one was checked).
- **Change.**
  1. `read_head_direction` raises (or warns, naming both readings) when a
     degrees-labelled series has every finite value within ±π and uses most of
     the circle.
  2. The directional entry points warn when headings cover little of the
     circle. Calibration on the clean recordings (36 direction bins, a bin
     counts if it holds ≥ 0.5 % of samples): Mouse32, 002030 and 001701 use
     36/36 bins; the misread 001701 headings use 2/36. A threshold of fewer than
     half the bins separates them. Not yet calibrated on a linear track, where
     headings cluster in two opposite directions; the next batch includes one
     (W-track, below) before this check lands.
- **Module.** `io/nwb` (reader), `encoding/directional` (check).
- **Changes results.** Only for files that are mislabelled today, which are
  currently wrong.

### 2. `detect_laps` counts laps where there is no loop (G-015)

- **Problem.** The loop test treats any cycle in the bin graph as a loop.
- **Evidence.** Three-bin cycles at the 63 junctions of the binary-tree
  labyrinth give 3 "laps"; grid neighbours give one "lap" in the 000582 box and
  one in the 002030 arena; the shrunken 000405 arena gives 3. Four datasets.
- **Change.** Test for a loop on the track (or ignore cycles confined to one
  junction or to grid neighbours). Taken by the UX-fixes plan, phase 6.
- **Module.** `behavior`. **Changes results.** Yes: spurious laps disappear.

### 3. Grid scores disagree with opexebo because of the autocorrelogram (G-006)

- **Problem.** On the same unsmoothed rate map, neurospatial's grid score can be
  far below the Moser-lab tool's, enough to change which cells pass a cutoff.
- **Evidence.** Reference check 2 (000582): unit 2 scores 0.179 vs 0.560; max
  difference 0.381, r = 0.737. Scoring agrees (neurospatial's scorer on
  opexebo's autocorrelogram: max difference 0.012); the autocorrelograms differ
  in two conventions: opexebo zero-fills unvisited bins (zero-filling moves
  unit 2 to 0.307), and neurospatial leaves undefined the lags where fewer than
  a quarter of visited bins overlap (2822 of 5041 lags). One dataset with a
  reference disagreement is the plan's bar.
- **Needs a design decision.** (a) Match BNT/opexebo (zero-fill and full
  lags) as the default, keeping the stricter rule as an option; (b) keep the
  NaN-respecting autocorrelogram but relax the overlap rule; (c) keep both and
  document the difference with this comparison. Before choosing, repeat the
  check on a dataset with many grid cells (Gardner 2022, next batch).
- **Module.** `encoding/grid`. **Changes results.** Yes under (a) or (b).

### 4. Spatial information on sparse occupancy (G-022)

- **Problem.** With little time per bin, spatial information is dominated by
  sampling noise, and the bias estimate `classify()` uses does not reflect it.
- **Evidence.** Bat, 3-D flight: median 3.38 bits/spike; with spikes circularly
  shifted (4 seeded shifts) 3.55; only 34 % of units beat every shift (000582:
  all 6, figure-8: 94 %, 001701: 92 %). `spatial_information_bias` has a
  median of 0.37. 70 % of visited bins hold under 0.1 s. One dataset, because
  it is the only one with sparse occupancy; the shift operation on the other
  three gives the contrast.
- **Needs a design decision.** First test `spatial_information_bias` against a
  shuffle null on sparse occupancy (simulated and the bat). Then either correct
  the estimate, or warn when most visited bins hold very little time and point
  to `min_occupancy` or larger bins.
- **Module.** `encoding` (`_metrics`, `SpatialRatesResult.classify`).
  **Changes results.** Classification changes if the estimate changes.

## Warning calibration

### Too aggressive

Only two neurospatial warning messages fired on correctly loaded data, both
the spike-drop warning when tracking covers part of a long recording (Mouse32
sleep spikes, figure-8 sleep spikes): a false alarm whose advice ("check that
spike_times and times share units") is wrong there, and a second copy of it
inside the same `decode.position` call (G-001). Both are in the UX-fixes plan
(phase 6: count only spikes inside the requested epochs, one discarded-data
warning per call). No threshold needs loosening. The only other warning was
hdmf's own deprecation notice while reading a 2022 file.

### Not aggressive enough

| Defect put back | Probes | Caught (incl. upstream) | Silent | No effect | Undetectable |
| --- | --- | --- | --- | --- | --- |
| Length unit off by 100 | 23 | 16 | 5 | 2 | 0 |
| Missing positions stored as 0 | 18 | 0 | 10 | 5 | 3 |
| Spike times in ms | 6 | 4 | 2 | 0 | 0 |
| Headings in degrees | 2 | 2 | 0 | 0 | 0 |
| Headings read as degrees (radians shrunk) | 1 | 0 | 1 | 0 | 0 |
| Positions read as mm (÷10) | 3 | 0 | 1 | 0 | 2 |
| Mirrored frame (image y down) | 5 | 0 | 0 | 4 | 1 |
| **Total** | **58** | **22** | **19** | **11** | **6** |

Proposed checks, each with its calibration on the clean recordings (a new
warning must fire on none of them) and on the hazards:

1. **Missing positions stored as 0 (G-017).** Warn when at least 0.1 % of
   tracked samples sit exactly at the origin. Clean: 0 of 7 recordings with
   positions have any sample exactly at the origin. Hazards: catches the
   labyrinth (4.9 %), 001701 (2.6 %) and the bat (0.24 %); misses 000582 (one
   sample, harmless). A broader rule ("many samples at one exact coordinate")
   would misfire: the clean bat recording has 15 % of samples at one perch
   position. Module: the shared position checks behind `Environment.from_samples`,
   the rate functions and `segment_by_velocity`.
2. **Length unit off by 100 (G-014).** Warn when the environment has fewer than
   5 bins along a dimension. Clean: the fewest is 10 (bat height). Hazards:
   every ÷100 case has 1–2 bins per dimension; today's check only fires at
   ≤ 1 bin and missed the figure-8 (2 bins). The ×100 case is already refused
   (bat, 9 × 10⁹ bins). Speed thresholds hide the same error in
   `segment_by_velocity` and the occupancy speed gate (668 vs 69 segments on
   the bat; 14 s vs 522 s occupancy on the figure-8); naming the track length
   in the environment repr would make it visible. Module: `environment`.
3. **Spike times in ms (G-010).** Compare the spike span with the span of the
   requested epochs (or the whole tracking) rather than with `times` alone.
   Today the clock-ratio hint catches it when tracking covers the recording
   (000582, bat, 001701) and misses it when tracking covers part (Mouse32,
   figure-8: rates drop to 0 with only the usual warning). Depends on G-001's
   fix. Module: `encoding/_spatial_counts`.
4. **Replay on events too short for the score (G-018).** Warn when an event
   has fewer time bins than a recipe's score needs, naming the recipe's event
   rule (both `DAVIDSON_2009` and `GROSMARK_2016` say ≥ 100 ms). Figure-8 awake
   ripples: 48 of 71 are one 20 ms bin. This warning should fire on that clean
   data: the events are too short. Module: `replay`.
5. **Headings squeezed into a small arc (G-031)** — see Wrong results 1.

Undetectable by values, so the answer is documentation: a mirrored frame flips
every left and right turn while the share of left turns stays near 0.5
(G-020); positions read as mm look like a plausible small arena (000405 ÷10).

## Missing capabilities

### 5. Spatial maps from a continuous signal (G-029)

- **Problem.** No public function makes a rate map from ΔF/F, deconvolved
  activity or binned rates; all take spike times.
- **Evidence.** 002030 (ΔF/F, 26 cells) and 000405 (binned rates, 46 cells)
  needed harness workarounds: an occupancy-weighted mean per bin, and transient
  onsets passed as spike times. The onset rule needs thresholds the library
  cannot see and saturates at high event rates.
- **Change.** A value-weighted rate map: occupancy-normalized mean of a signal
  per bin, sharing the rate maps' bins, NaN rules, smoothing and result type;
  plus a documented ΔF/F recipe. Deconvolution stays out of scope.
- **Module.** `encoding`. **Changes results.** No (new entry point).

### 6. NWB readers for data outside the standard places (G-011, G-012, G-032)

- **Problem.** Real files keep intervals, head direction and tracking where the
  readers do not look.
- **Evidence.** 000552: ripples and sleep states are `TimeIntervals` in
  processing modules, which `read_intervals` does not search (G-011); the
  `Position` container's name is rejected as `position_name` (G-012). 002030:
  head direction is a bare `SpatialSeries` (G-032). 000405: position and head
  direction live in one generic `TimeSeries` with columns named in its
  description (G-032). Three datasets.
- **Change.** Search processing modules in `read_intervals`; accept a container
  name when it holds one series; accept an explicit path to any `SpatialSeries`
  or `TimeSeries` (with column choice) in the position and head-direction
  readers.
- **Module.** `io/nwb`. **Changes results.** No.

### 7. Decoding head direction (G-004)

- **Problem.** There is no public heading decoder.
- **Evidence.** Mouse32: pynapple's `decode_bayes` on the 31 ADn units reaches
  8.9° median error (outside tracking gaps), reproducing its tutorial's 19,590
  samples. A reference result neurospatial cannot produce.
- **Needs a design decision.** A circular (ring) environment the existing
  decoder accepts, or a documented recipe.
- **Module.** `decoding`, `environment`. **Changes results.** No.

### 8. Periodic 1-D tracks (G-021, G-026)

- **Problem.** A linearized loop, whose two ends are the same place, is treated
  as a line.
- **Evidence.** Figure-8: 0 and 222 cm sit at the same camera point; 62 wraps
  read as jumps of about 216 cm, 57 of them inside running segments as speed
  spikes of about 6,500 cm/s; laps cannot be counted; decoding and replay
  treat the ends as 222 cm apart. One dataset, but loop and circular tracks
  are common (000044 includes a 1 m circular track).
- **Change.** A periodic 1-D track (a length and a flag that the ends meet), or
  a recipe that embeds the loop as a circle for `from_graph`.
- **Module.** `environment`, `behavior`. **Changes results.** No.

## Friction and error messages

- **Declared geometry before boundary measures (G-005, G-016).** Border scores
  refuse an environment built from samples (000582, 001701), and a track graph
  without `track_width` cannot place positions (labyrinth). Both refusals are
  right and their messages say what to do; show the declaration in the
  quickstart and the `from_graph` example. Module: docs.
- **`Position` container name (G-012)** — see 6.
- **Grid and border scores on 3-D (G-024)** — refused with clear messages;
  won't fix.

## Docs

- **Spike-to-sample rule (G-003).** neurospatial assigns a spike to the sample
  at or before it, pynapple to the nearest; on 000582 this moves 218 of 4013
  spikes to another bin and spatial information by up to 2.1 %. State it with
  the rate maps.
- **Decoding at chance (G-007, G-023).** Six MEC units decode at chance (66 cm
  vs 55 cm), and silent bins all decode to the lowest-rate corner; the bat
  decodes near chance (2.18 m vs 2.34 m). Report a chance level next to
  `error_against`, or point to `posterior_entropy`.
- **Turn directions assume y up (G-020).** Say so in `graph_turn_sequence`.
- **`DAVIDSON_2009` verdicts depend on the batch (G-027).** The pseudo-event
  null draws from the events passed together: the same stationary events are
  called replay 7/8 times with sweeps and 0/16 alone. Stationary events can be
  replay; the point is the batch dependence. Say it in the recipe and the
  replay guide.

## Performance

Nothing to fix at these sizes. The slowest operation was the replay comparison
on 71 events with published shuffle counts (14 s); the largest traced peak was
265 MB (bat, four extra rate maps); the bat run peaked at 1.3 GB resident.
Long recordings with thousands of replay events (the 4,226 figure-8 sleep
ripples) are the next scale to measure.

## Reference-check verdicts

| Check | Verdict |
| --- | --- |
| 1. Spatial information vs pynapple (000582) | Agree by convention: within 2.1 %, all from the spike-to-sample rule (G-003). The default `binned` map is now unsmoothed (G-002, fixed) |
| 2. Grid score vs opexebo (000582) | Disagree, needs a decision (proposal 3) |
| 3. Head-direction decoding vs pynapple (Mouse32) | neurospatial cannot do it (proposal 7); pynapple's 8.9° is the target |

## Data quirks: detect, or leave to the loader

| Gap | Quirk | neurospatial should |
| --- | --- | --- |
| G-030 | Radians labelled degrees | Detect (proposal 1) |
| G-026 | Linearized loop wraps around | Support it (proposal 8) |
| G-028 | Behavior starts before the neural recording | Leave to `spike_window=` when the recording span is known; nothing warned about the bat's 1.35 s of flight before the first spike |
| G-025 | Positions outside the stated room bounds | Leave to the loader; only an environment with declared bounds could flag them |
| G-008 | Head-direction frame not stated | Return the empty frame from `read_head_direction` (it returns None), so callers can see it is unknown |
| G-033 | Labels disagree with values (mm, period as rate, "uv") | Leave to the loader: these are generic `TimeSeries` the readers do not read |
| G-009, G-034, G-035 | One LED; ΔF/F scale; inverted LFP conversion | Nothing |
| G-019, G-036 | Pre-registered expectations that were wrong | Nothing (kept as registered) |

## Should loader code move into neurospatial?

No, except through the readers. The eight loaders total about 870 lines of
code; of that, roughly 15–20 lines each are the pinned download and the
standalone summary, and the rest is specific to one file: column names, unit
fixes, epoch selection, a restricted unpickler for one pickled dataset, a
MATLAB-table reader for one zip. The generic parts that matter to users are
the NWB reader gaps in proposal 6 and the unit checks in proposal 1, which
belong in `io/nwb`. A `neurospatial.datasets` fetcher (pooch-style) is only
worth it if tutorials come to depend on three or more public recordings; the
harness's `_fetch.py` (77 lines) would be the starting point.

## Next batch

Chosen to fill the weakest cells of the matrix: operations exercised on only
one recording, or never.

| Dataset | Fills |
| --- | --- |
| Gardner 2022 grid modules (Figshare 16764508, CC0) | `cells.grid_score` at scale for proposal 3 (hundreds of grid cells vs six). Needs a decision: its `.npz` files require `allow_pickle`; the restricted-unpickler approach used for the labyrinth would apply |
| Shin & Jadhav W-track (DANDI 000447, already used in the project) | A real track graph with arms and wells: `env.from_graph`, `behavior.laps`, `behavior.turn_sequence` with spikes; a linear-track heading distribution to calibrate proposal 1's check; its velocity-as-third-column quirk |
| Plitt & Giocomo VR imaging (DANDI 000054) | `deconvolved` activity (never run), the −500 position sentinel (a new hazard), 1-D VR tracks |
| Grieves 2020 climbing lattice (UCL RDR, CC0, 11.2 GB zip) | A second 3-D dataset to test G-022 and G-023 off the bat; a lattice-shaped 3-D graph. Needs `inflate64` for its Deflate64 zip |
| Zhang 2022 egocentric boundary cells (Figshare 21647423, CC BY) | Egocentric operations, never exercised; MATLAB `sdata` structs |
| Grosmark & Buzsáki (DANDI 000044, already used) | Replay on long sleep candidate events (G-018 at scale), circular track (proposal 8), the 4-hour runtime case |

## Tutorial candidates

- `load_sargolini_000582.py`: the shortest real "bring your own NWB" example.
  Standard containers, the readers work as they are, and a published reference
  table to compare with.
- `load_giocomo_001701.py`, once proposal 1 lands: a real recording that shows
  why to check value ranges against unit labels, with accurate decoding and
  head-direction tuning.

## Gap index

| Gap | Where it is addressed |
| --- | --- |
| G-001, G-010 | Warning calibration (too aggressive; check 3); G-001 in UX-fixes phase 6 |
| G-002 | Fixed (UX-fixes phase 3b) |
| G-003 | Docs |
| G-004 | Proposal 7 |
| G-005, G-016 | Friction |
| G-006 | Proposal 3 |
| G-007, G-023 | Docs |
| G-008, G-009, G-019, G-025, G-028, G-033–G-036 | Data quirks |
| G-011, G-012, G-032 | Proposal 6 |
| G-013 | Fixed (UX-fixes phase 3b) |
| G-014 | Warning calibration, check 2 |
| G-015 | Proposal 2 (UX-fixes phase 6) |
| G-017 | Warning calibration, check 1 |
| G-018 | Warning calibration, check 4 |
| G-020 | Docs; undetectable |
| G-021, G-026 | Proposal 8 |
| G-022 | Proposal 4 |
| G-024 | Won't fix (behaves as documented) |
| G-027 | Docs |
| G-029 | Proposal 5 |
| G-030, G-031 | Proposal 1 |

## Decisions for the maintainer

1. **Head-direction label check (proposal 1):** ship the reader refusal now as
   its own PR, and the coverage warning after the W-track calibration?
2. **Grid score (proposal 3):** match BNT/opexebo, relax the overlap rule, or
   document the difference — and wait for the Gardner comparison first?
3. **Spatial information bias (proposal 4):** run the shuffle-null test of
   `spatial_information_bias` before deciding between correcting it and
   warning on sparse occupancy?
4. **Warning checks 1, 2 and 4:** add them as calibrated here (origin share
   ≥ 0.1 %, fewer than 5 bins per dimension, events shorter than the score
   needs)? Check 3 waits for G-001's fix.
5. **New capabilities:** which of continuous-signal rate maps (5), NWB reader
   coverage (6), heading decoding (7) and periodic tracks (8) to plan, and in
   what order?
6. **Next batch:** run the six datasets above, and accept the restricted
   unpickler for Gardner 2022?
7. **Tutorial:** promote `load_sargolini_000582.py` to `examples/` now?
