# Real-data survey: findings

Thirty-seven public recordings were loaded into plain arrays and run through
21 operations (20 calls into neurospatial and one harness workaround), each
recording was re-run with known defects put back (332 probes), and results
were compared with pynapple, opexebo and the authors' own grid scores. This
document ranks what that showed into proposed changes to neurospatial. The
evidence lives in [GAPS.md](GAPS.md) (gap entries `G-###`) and
[results/](results/) (`matrix.md`, `warnings.md`, `reference_checks.md`, one
JSON per dataset, and `studies/` for the calibration and information-bias
studies). A few early calibration numbers were computed directly from the
loaders (marked *loader-computed*); the warning numbers are now produced by
`studies/warning_calibration.py`.

First batch: DANDI 000582 (rat MEC open field), OSF Mouse32 (mouse ADn head
direction), DANDI 000552 (mouse CA1 figure-8, ripples, sleep), the Rosenberg
2021 labyrinth (behavior only, binary-tree maze), the Yartsev 2025 bat demo
(3-D flight), DANDI 002030 (mouse CA1 miniscope ΔF/F), DANDI 000405 (rat open
field, binned rates, mislabelled tracking) and DANDI 001701 (mouse MEC/CA1 on an
X maze, mislabelled head direction).

Second batch, chosen for the decisions below: Gardner 2022 (rat MEC, three grid
modules, 483 units), Vollan 2025 (rat MEC Neuropixels: rat 27764 on a linear
track with tracked head direction, rat 26018 in an open field with three grid
modules and the authors' grid scores), DANDI 000044 (rat CA1 on a 1 m circular
track and a 1.6 m linear track, with hours of sleep; read by HTTP byte range
from 9 GB files) and DANDI 000447 (rat CA1/PFC on a W-track, also streamed).

Third batch, for breadth (details in [BATCH_NWB_BREADTH.md](BATCH_NWB_BREADTH.md)):
nine small NWB recordings from DANDI 000987, 001880, 001695, 001754, 000568,
000165, 001845, 000003 (streamed) and EBRAINS (Kanter 2025), chosen for the
ways real files store position, time and units: per-axis series, blank units,
`conversion` factors, pixels, position sampled at spike times, pose, two
probes in one units table.

Fourth batch, non-NWB archives (details in [BATCH_ARCHIVES.md](BATCH_ARCHIVES.md)):
the Grieves 2020 3-D climbing lattice (a Deflate64 zip member read by byte
range), the Morris water maze (EthoVision JSON, behavior only), Ólafsdóttir
2016 (Axona files parsed by hand), Takigawa 2022 (two tracks, two pixel
scales), Hafting 2008 (NIRD), J16 (the authors' track graph), Agarwal 2023 (a
3-arm maze, 439 units) and Shipley 2025 (8-arm radial maze, NPY). Zhang 2022's
egocentric boundary deposit was dropped: it has no per-sample tracking
(G-070).

Fifth batch, CRCNS sets behind an account (details in [BATCH_CRCNS.md](BATCH_CRCNS.md)):
hc-11 Achilles 25 Oct 2013 (the same session as 000044's linear track), hc-3
(a 250 cm linear track, two LEDs, Neuroscope `.whl`/`.res`/`.clu`), hc-2 (a
180 cm open field), hc-28 (a W-track whose `rawpos` stores 0 for untracked
frames), hc-31 (a 48 m track in inches and µs) and hc-18 (a model train,
32,552 Hz clock). hc-13 is one 8.1 GB tarball and was not loaded.

Declared environments (details in [ENVIRONMENT_FIDELITY.md](ENVIRONMENT_FIDELITY.md),
[BATCH_FAITHFUL_ARENAS.md](BATCH_FAITHFUL_ARENAS.md) and
[BATCH_FAITHFUL_TRACKS.md](BATCH_FAITHFUL_TRACKS.md)): 30 loaders now declare
the apparatus a source states (an outline, a room, a track graph, the
lattice's bars), and every recording is run twice, on the declared
environment (`results/`) and on one built from the samples
(`results/inferred_env/`, compared in `results/studies/declared_vs_inferred.md`).
000447's two W-tracks are now separate recordings. The counts in this
introduction and in "At a glance" (recordings, probes, silent defects) are
from the first five batches; the full rerun after this batch refreshes them.

## At a glance

- **The common path holds up.** Environments, occupancy, rate maps, spatial
  information, decoding, NWB round trips and behavior segmentation ran on every
  recording that had the inputs, in 1-D, 2-D and 3-D, including 483 units and
  9 GB files read by byte range. Decoding is accurate where the data allow it
  (figure-8 8.0 cm, 001701 3.5 cm median error).
- **Warnings are not too aggressive.** Across the fourteen clean recordings no
  neurospatial warning fires that should not. The masked-data warning fires on
  Vollan, whose authors did remove slow samples; it is right there.
- **Warnings are not aggressive enough.** Of 332 probes, 139 defects passed
  silently and changed the result. Four checks calibrated below
  (exact-origin samples on 2-D/3-D positions, too few bins along every axis,
  squeezed headings, a speed gate that removes every sample) fire on none of
  the clean recordings and would catch 92 of the 139, including a real file's
  own convention (hc-28 stores 0 for untracked frames: 15.5 % at the origin). The largest class nothing
  catches is a missed pixel calibration (12 probes, G-080): it makes the arena
  look 2–4× too big, which is plausible.
- **The same session, two archives, one answer.** CRCNS hc-11 and DANDI
  000044 give the same 137 units, all 8,414,607 spike times exactly, the
  same 80,762 position samples within 5e-8 s and the same epochs; only the
  sleep scoring differs (5 states vs 3), which moves the replay rate from
  16 % to 14 % (G-090).
- **Labels lie often.** Four of the nine third-batch files mislabel position
  in four different ways (blank unit, a `conversion` that contradicts the
  values, pixels labelled metres, pixels labelled cm); in every case a
  magnitude check against the apparatus decided, never the label.
- **Three wrong results need fixing first**: `read_head_direction` turns
  radians labelled "degrees" into a 6° arc (G-030/G-031); `detect_laps` finds
  laps where there is no loop (G-015) and misses them where there is one: 1
  lap where a rat ran about 100 laps of a circular track (G-037); and grid
  scores collapse for large-spacing grid modules (G-006; the data point to a
  fix, below).
- **The spatial-information bias estimate is right for what it states.** On
  every recording, sparse 3-D occupancy included, it matches simulated untuned
  Poisson cells; bursty cells and real units sit 2–7× above it. A
  sparse-occupancy warning would be the wrong fix (proposal 4).
- **Declared environments are right where the samples-built ones are not.**
  8 of the 10 declared arenas and 3-D environments match (stated size
  within 15 %), against 2 of 10 built from the samples; the bat's room is
  a mismatch, now that a room is judged by its faces: 17 of the 57 flight
  samples nearest one wall lie 1.4 m past it, at the perch beyond `r_lim`
  (G-025). The lattice is not verified: its side matches (+0.0 % of
  97 cm), but judged as a whole its head samples sit 9.4 cm horizontally
  off the authors' bars by the refitted shift (2.35 bins) and 0.38 bin by
  a single step, which on a synthetic lattice reads under a bin for shifts
  up to about 2.5 bins (G-159). Of 20 declared tracks, 11 match and 9 are
  not verified (no fact independent of the samples, a frame offset that
  depends on the estimator, or corridors closer than twice the window the
  offset is read in), against 4 matches, 7 mismatches and
  9 not verified built from the samples. Every declared track and the
  lattice is now also judged as a whole: one translation fitted to every
  straight piece's median offset, any length, read by a single
  least-squares step and refitted until it settles; both over one bin is a
  mismatch, straddling it not verified. The
  labyrinth and Takigawa's tracks, which have no stretch of the 20 bins a
  stretch's bias needs, now have a whole-track check. Takigawa's passes
  (0.40 bin); the labyrinth's reads 0.21 bin over 127 pieces but its
  corridors lie 1 cell apart, so a shift past half a cell would read as
  the next corridor's, and it is not verified (G-160). At the real
  recordings' scatter the single step stays under a bin, so this check
  gives at most "not verified" on a misplaced layout. On 000044's linear track the two
  straddle the limit (0.50 and 1.18 bins across the track), on a lateral
  scatter far wider than the 3-bin window the offset is read in, so it is
  not verified (G-160). Samples cannot show walls thinner than a bin (the
  labyrinth's corridors join, G-110) or tell apart two apparatus in one
  camera frame (G-113). Head samples beyond a track's width or short of a
  wall are expected (headstage LEDs, swings, camera projection), so they
  are reported, not judged (G-152, G-157).
- **What declaring changes.** Laps exist on the declared loops (000044's
  circle 67 and hc-18 35, against 1 each, G-037), but a sample outside the
  track splits the recording and costs laps (G-151). Border scores run on
  declared arenas, and are -1 for most cells where the tracked head stays a
  few cm off the wall (407 of 619 on Vollan's open field, G-131). Grid
  scores of grid cells do not change (Gardner median 0.953 against 0.949).
  Decoding errors fell where train and test now alternate within each arm
  instead of by even and odd trial, which split them by direction or loop:
  000447's familiar W-track 35.9 to 4.9 cm, the figure-8 7.8 to 5.9 cm.
- **Biggest missing capability**: spatial maps from calcium imaging or binned
  rates (G-029), deferred; loop tracks (G-021), now on the plan.

Each proposal below states the problem, the evidence, the change, the module,
and whether results change for existing users. Proposals are ordered by
scientific risk, then by how many datasets showed the problem. Where one
dataset is the evidence, the section says why that is enough.

## Wrong results

### 1. Warn on a "degrees" label on head direction that is radians (G-030, G-031)

- **Problem.** `read_head_direction` trusts a series' unit label, so radians
  labelled "degrees" come back as a 6° arc, and downstream nothing notices.
- **Evidence.** 001701: values −3.140 to 3.142 labelled degrees; the reader
  returns −0.055 to 0.055 rad. `compute_directional_rates` on those headings
  gives a median mean vector length of 0.999 over 320 units (true 0.12): every
  unit looks like a sharp head-direction cell. One dataset is enough: the
  conversion is deterministic, so every file with this mislabel is wrong in the
  same way (001701 has 218 files; only one was checked).
- **Change.**
  1. `read_head_direction` warns, naming both readings, when a
     degrees-labelled series has every finite value within ±π and uses most of
     the circle (maintainer decision 1: warn, not refuse).
  2. The directional entry points warn when headings cover little of the
     circle. Calibration on the clean recordings (*loader-computed*: 36
     direction bins, a bin counts if it holds ≥ 0.5 % of samples): Mouse32,
     002030 and 001701 use 36/36 bins; the misread 001701 headings use 2/36. A
     threshold of fewer than half the bins separates them. On a linear track
     (Vollan rat 27764, OptiTrack head direction), where running clusters in
     two opposite directions, it is still 36/36 over all samples and over
     running samples, as are Gardner and Vollan's open field
     (`results/studies/warning_calibration.md`).
- **Module.** `io/nwb` (reader), `encoding/directional` (check).
- **Changes results.** Only for files that are mislabelled today, which are
  currently wrong.

### 2. `detect_laps` counts laps where there is no loop, and misses them on one (G-015, G-037)

- **Problem.** Before #265 the loop test treated any cycle in the bin graph as
  a loop. #265 tests the track graph's cycles or holes in a 2-D area instead.
- **Evidence.** Before #265: three-bin cycles at the 63 junctions of the
  binary-tree labyrinth gave 3 "laps"; grid neighbours gave one "lap" in the
  000582 box and one in the 002030 arena; the shrunken 000405 arena gave 3.
  After #265 the labyrinth, the 001701 maze and the 3-D bat space are refused,
  as they should be. Two open arenas still give laps: one clockwise in the
  000582 box and one counter-clockwise in the 002030 arena (000405 returns
  0 laps rather than a refusal). An environment built from samples leaves
  patches of unvisited bins inside the arena, and these count as holes to run
  around (*loader-computed*, approximate: 72 enclosed patches in 000582, 105
  in 000405, 19 in 002030 including a 412-bin unvisited centre, where the
  mouse kept to the wall).
- **CRCNS.** A second real loop fails the same way: on the hc-18 model
  train's circuit (about 107 laps) `detect_laps` finds 1. hc-2's open field
  gives 1 clockwise lap; the hc-28 W-track is refused.
- **Second batch.** Gardner's 1.5 m box is refused, Vollan's box gives 0 laps,
  and the 000447 W-track (no loop) gives 1. On a real loop the opposite
  fails: on the 000044 circular track the rat's angle around the centre turns
  97 times net and the linearized coordinate wraps 123 times, but
  `detect_laps` returns 1 lap with its defaults, 34 with `min_overlap=0.5`,
  and 0 with `max_gap=None` (G-037). 18.5 % of samples are untracked.
- **Needs a decision.** Whether running around an unvisited part of an open
  arena counts as a lap. If not, holes should come from the declared arena
  (walls, obstacles), not from where the animal happened not to go.
- **Module.** `behavior`. **Changes results.** Yes: spurious laps disappear.

### 3. Grid scores collapse for large-spacing modules (G-006)

- **Problem.** neurospatial's autocorrelogram leaves undefined every lag where
  fewer than a quarter of the visited bins overlap. For a module with large
  grid spacing, the outer ring of peaks that the grid score reads lies at
  those lags, so the score collapses, enough to change which cells pass a
  cutoff.
- **Evidence.** 000582 (6 units): up to 0.38 below opexebo on the same map.
  Populations (`reference_checks.md` 2b–2c; 5 cm smoothed maps on a full
  2.5 cm grid; every convention scored with neurospatial's `grid_score`
  except opexebo's own; verdicts at a conventional 0.3 cutoff, with an
  undefined score counted as a different verdict):

  | Convention | Gardner, 483 units: r vs opexebo / verdicts differing | Vollan, 233 units: r vs opexebo / verdicts differing | Vollan module 1 / 2 / 3 median (spacing 46 / 65 / 93 cm; authors 1.11 / 0.99 / 0.68) |
  | --- | --- | --- | --- |
  | today (quarter overlap) | 0.886 / 56 | 0.775 / 42 | 1.19 / 0.98 / **0.16** |
  | (a) every lag with ≥ 20 overlapping bins | **0.966 / 22** | 0.828 / 41 | 1.19 / 1.01 / **0.77** |
  | (c) zero-fill unvisited bins | 0.933 / 34 | **0.994 / 2** | 1.03 / 0.63 / 0.44 |
  | opexebo (zero-fills) | 1 / 0 | 1 / 0 | 1.03 / 0.63 / 0.48 |

  Against the authors' own scores (their maps and scorer, module medians
  only): module 1 all conventions are 0.08 off; module 2 today's rule is
  closest (0.01 off; (a) 0.02; zero-filling 0.36); module 3, the largest
  spacing, today's rule collapses to 0.16 and (a) is closest (0.77 vs 0.68).
  Against opexebo, (a) agrees best on Gardner, but on Vollan zero-filling
  agrees almost exactly because opexebo zero-fills, and (a) is barely closer
  than today's rule. The quarter-overlap rule over the full lag range
  reproduces today's scores exactly, which checks the harness version. (b), a
  relaxed overlap rule, is (a) with another threshold.
- **Recommendation.** (a): a NaN-aware Pearson correlation at every lag with
  at least about 20 overlapping visited bins (`nan_aware_autocorrelation` in
  `reference_checks.py`, tested against a direct per-lag computation,
  including constant overlaps). It keeps "missing is not zero", removes
  today's collapse on the large-spacing module, and stays within 0.02 of the
  authors on the other two. Matching opexebo exactly would need zero-filling,
  which moves module 2 from the authors' 0.99 to 0.63. Limits: three module
  medians, not per-unit author scores, and one cutoff not taken from either
  paper.
- **Module.** `encoding/grid`. **Changes results.** Yes: large-spacing scores
  rise; 22 of 483 Gardner verdicts still differ from opexebo.

### 4. Spatial information on sparse occupancy (G-022)

- **Problem.** On the bat's sparse 3-D occupancy, 15 % of units pass the bias
  screen (information minus `spatial_information_bias()` above 0.1 bits/spike)
  while their information is below the 95th percentile of their own circular
  shifts. The question was whether sparse occupancy breaks the bias estimate.
- **Evidence** (`results/studies/information_bias_null.md`; 200 simulated
  units per row, 100 shifts per real unit, seeded):
  - Untuned Poisson cells on each recording's own occupancy: mean information
    / mean bias is 0.92–1.02 on every recording, the bat (70 % of visited bins
    under 0.1 s) included. Where a unit has too few spikes for the bias (the
    bat at 0.5 and 2 Hz), it is NaN, not wrong.
  - Untuned bursty cells (bursts of mean 3 spikes) at the same rates: 2.2–5.0×
    the bias, close to the factor of 5 a geometric burst size of mean 3
    predicts (E[k²]/E[k]).
  - Real units, with each unit's spikes shifted within the time the maps
    count: the shift null's median is 1.23× the bias on the bat and 6.4–7.3×
    on the dense recordings (000582, Gardner, 000044). Only on the bat does it
    change a verdict: everywhere else every unit that passes the bias screen
    also beats its shifts. On Vollan's speed-filtered data 84 % (open field)
    and 86 % (linear track) of units beat every shift; the earlier result,
    with real units below their shifts, was a harness bug that shifted
    spikes out of counted time (G-043, resolved).
- **Conclusion.** The estimate is right for what it states, the information
  of an untuned Poisson cell. The gap is temporal structure, mainly bursts,
  which circular shifts keep and a Poisson null does not (the shifts do not
  keep firing locked to a flight's or a run's start). It matters only when real information is close to that null, as
  on the bat. A sparse-occupancy warning would be the wrong fix (the agreed
  warning plan does not apply). A second sparse 3-D recording agrees (the
  Grieves lattice: Poisson ratios 1.00–1.01, bursty 3.3–4.7, real units'
  shift null 2.56× the bias, 6 % pass the bias screen but fail the shifts).
- **Recommendation.** Say in `spatial_information_bias` and in
  `classify()`/`is_*_cell()` (threshold screens) that the bias is a Poisson
  null that bursty firing exceeds, and point to the shuffle verdicts
  (`criterion="shuffle"`, already the default of the free predicates) when
  information is low. No change to the estimate.
- **Module.** `encoding` docs. **Changes results.** No.

## Warning calibration

### Too aggressive

Before #265, two neurospatial warning messages fired on correctly loaded data,
both the spike-drop warning when tracking covers part of a long recording
(Mouse32 sleep spikes, figure-8 sleep spikes): a false alarm whose advice
("check that spike_times and times share units") was wrong there, and a second
copy of it inside the same `decode.position` call (G-001). #265 counts only
spikes inside the requested epochs and warns once per call; on the current
`main` no neurospatial warning fires that should not on any of the fourteen
clean recordings. No threshold needs loosening. One warning fires on clean data
and is right: the masked-data warning on both Vollan recordings ("about 6 % of
the analysed time is in runs of removed samples … looks like times were
boolean-masked"); the authors did remove every sample under 5 cm/s, and the
message's own fix (max_gap below the shortest removed run) applies. The only
other warning is hdmf's deprecation notice while reading a 2022 file.

### Not aggressive enough

| Defect put back | Probes | Caught (incl. upstream) | Silent | No effect | Undetectable |
| --- | --- | --- | --- | --- | --- |
| Length unit off by 100 | 107 | 84 | 22 | 0 | 1 |
| Missing positions stored as 0 | 72 | 0 | 66 | 3 | 3 |
| Spike times in ms | 40 | 38 | 2 | 0 | 0 |
| Pixels taken for cm (calibration missed) | 12 | 0 | 12 | 0 | 0 |
| Headings in degrees | 9 | 9 | 0 | 0 | 0 |
| `.whl` −1 (untracked) kept as a coordinate (hc-3, hc-2) | 9 | 0 | 9 | 0 | 0 |
| Times rebuilt from `dt` over speed-filter gaps (Vollan) | 9 | 4 | 3 | 2 | 0 |
| Mirrored frame (image y down) | 8 | 0 | 0 | 6 | 2 |
| Speed column read as a third position axis (000447) | 5 | 1 | 4 | 0 | 0 |
| Untracked samples stored as (0, 0) (Agarwal) | 4 | 0 | 4 | 0 | 0 |
| `conversion` applied to values already in cm (001695) | 4 | 3 | 1 | 0 | 0 |
| Epoch table left transposed (hc-31) | 4 | 2 | 0 | 2 | 0 |
| `pos` −1 (untracked) kept as a coordinate (hc-18) | 4 | 0 | 4 | 0 | 0 |
| Inches read as cm (hc-31) | 4 | 0 | 0 | 0 | 4 |
| Times left in µs (hc-31) | 4 | 3 | 0 | 1 | 0 |
| Clock from the header's nominal rate (Shipley) | 4 | 0 | 3 | 1 | 0 |
| Video clock instead of position clock (000568) | 4 | 0 | 0 | 3 | 1 |
| Duplicate timestamps kept (000165) | 3 | 2 | 0 | 1 | 0 |
| Tracking kept beyond the spike recording (000165) | 3 | 1 | 0 | 1 | 1 |
| Pixel keypoint labelled cm (001845) | 3 | 0 | 3 | 0 | 0 |
| Positions read as mm (÷10) | 3 | 0 | 0 | 1 | 2 |
| Lost-LED code (0, 255) kept (001754) | 3 | 0 | 2 | 1 | 0 |
| Vertical axis taken as floor (000987) | 3 | 0 | 0 | 0 | 3 |
| Coincident LEDs read as heading 0 (hc-3, hc-2) | 2 | 0 | 2 | 0 | 0 |
| Front and back LED swapped | 2 | 0 | 0 | 0 | 2 |
| Units merged by repeated id (Kanter) | 2 | 0 | 0 | 0 | 2 |
| Trials joined back to back (Shipley) | 2 | 0 | 0 | 2 | 0 |
| Headings stored in [0, 2π) | 1 | 0 | 0 | 1 | 0 |
| Headings read as degrees (radians shrunk) | 1 | 0 | 1 | 0 | 0 |
| Spike clock assumed 20 kHz for a 32,552 Hz file (hc-18) | 1 | 0 | 1 | 0 | 0 |
| **Total** | **332** | **147** | **139** | **25** | **21** |

Counts are for the `main` of the run. Probes are skipped when the operation
already fails on the clean recording.

Proposed checks, each with its hits on the clean recordings (a new warning
must fire on none of them) and on the hazards. Clean-recording numbers are in
`results/studies/warning_calibration.md` and the batch notes.

1. **Missing positions stored as 0 (G-017).** Warn when at least 0.1 % of
   tracked samples of a **2-D or 3-D** position sit exactly at the origin.
   Clean: none of the 36 recordings with positions has a single 2-D/3-D
   sample at the origin. Hazards: catches all 66 silent missing-as-zero
   probes and the 4 where untracked samples are stored as (0, 0) (Agarwal);
   on hc-28 with its documented zeros restored, 15.5 % of samples sit at the
   origin. The
   start-of-track case rules out 1-D input: on the 000044 circular track,
   0.23 % of the file's linearized samples are exactly 0, a real place. A
   lost-tracking code at another corner (001754's (0, 255)) is not seen. A
   broader rule ("many samples at one exact coordinate") would misfire: the
   clean bat recording has 15 % of samples at one perch position. Module: the
   shared position checks behind `Environment.from_samples`, the rate
   functions and `segment_by_velocity`.
2. **Length unit off by 100 (G-014).** Warn when the environment has fewer
   than 5 bins along **every** dimension (`env.grid_shape`). The accepted
   per-dimension rule misfires: 001695's 6 cm-wide linear track, recorded in
   2-D, has 4 bins across (the fewest along any axis of any clean recording;
   its long axis has dozens). With "every dimension", no clean recording
   fires; every ÷100 hazard has 2–3 bins along every axis (the 1-D
   figure-8: 3). Covers the figure-8's 3 environment-based silent probes
   (its `segment_by_velocity` probe falls to check 4). Module: `environment`.
3. **Headings squeezed into a small arc (G-031).** Warn when headings fill
   fewer than half of 36 direction bins (a bin counts at 0.5 % of samples).
   Clean: every recording with head direction fills 36 of 36 over all and over
   running samples (Vollan linear track, 000987, Kanter, J16's unused
   orientation included), except the hc-3 linear track: 30 of 36 (31
   running), the lowest of any recording, still well above 18. Hazard: the misread 001701 headings fill 2.
4. **A speed gate that removes every sample (G-014).** `segment_by_velocity`
   builds no environment, so positions ÷100 give 0 running segments without a
   word (000044 ×2, 000447, 001880, 001695 ×2, 000568, 001845 and six
   archive recordings). Warn when no sample exceeds `min_speed` while
   positions move. Catches 18 probes (the figure-8's, hc-11's, hc-28's,
   hc-31's and 001695's
   `conversion` case included); misses the bat's ×100 (668 segments instead
   of 69) and Ólafsdóttir ÷100 (668 → 1 segment of 0.58 s).
5. **Spike clock against the analysed span (G-010).** Since #265 the
   spike-drop count was limited to the requested epochs, so ms spike times
   passed silently when `epochs` were given (000165). Decided and merged
   (#272): spikes are treated as another clock when their span exceeds
   10^1.5× the tracked span and their largest value exceeds 10^1.5× every
   bound; on the survey's clean data the largest ratios are 17× span and 2×
   reach, ms data at least 307×. Accepted trade-off: a seconds recording more
   than 32× longer than tracking that starts with it (10 min at the start of
   6 h) also warns. Still open: 001695 in ms warns with the same text as its
   clean run, because its 5,800× ratio is outside the hint's 300–3,200×
   band.

Checks 1–4 would catch 92 of the 139 silent probes (check 5, if adopted, 2
more). The rest: a −1 untracked marker kept as a coordinate (13, G-096; a
constant sentinel other than the origin), pixels taken for cm (12, G-080),
the speed column read as a position axis (4, G-042), a clock from the
header's nominal rate 0.17 % off (3, G-081), times rebuilt over the speed
filter (3, G-040), a pixel keypoint labelled cm (3, G-056), the lost-LED code
(2, G-053), coincident LEDs read as heading 0 (2, G-094), a spike clock
assumed 20 kHz when the misplaced spikes fall outside the epochs (1, G-097),
and two `segment_by_velocity` cases that do not end at zero segments (the
bat ×100; Ólafsdóttir ÷100). A sentinel rule (many samples at one exact
coordinate on the edge of the tracked range) would cover G-053 and G-096 but
must spare the bat's perch (15 % of samples at one position). Undetectable by values, so
the answer is documentation: a mirrored frame (G-020), positions read as mm, a
video clock 23 s off the position clock (G-054), a vertical axis taken as
floor, units merged by repeated id (G-057).

The rule for replay events too short to score (G-018) is dropped (maintainer
decision 4).

## Missing capabilities

### 5. Spatial maps from a continuous signal (G-029)

- **Problem.** No public function makes a rate map from ΔF/F, deconvolved
  activity or binned rates; all take spike times.
- **Evidence.** 002030 (ΔF/F, 26 cells) and 000405 (binned rates, 46 cells)
  needed harness workarounds: an occupancy-weighted mean per bin, and transient
  onsets passed as spike times. The onset rule needs thresholds the library
  cannot see and saturates at high event rates.
- **Change.** A value-weighted rate map: occupancy-normalized mean of a signal
  per bin, sharing the rate maps' bins, NaN rules and smoothing, with the
  signal's own unit (ΔF/F is not Hz); plus a documented ΔF/F recipe.
  Deconvolution stays out of scope.
- **Module.** `encoding`. **Changes results.** No (new entry point).

### 6. NWB readers for data outside the standard places (G-011, G-012, G-032)

- **Problem.** Real files keep intervals, head direction and tracking where the
  readers do not look.
- **Evidence.** 000552: ripples and sleep states are `TimeIntervals` in
  processing modules, which `read_intervals` does not search (G-011); the
  `Position` container's name is rejected as `position_name` (G-012). 002030:
  head direction is a bare `SpatialSeries` (G-032). 000405: position and head
  direction live in one generic `TimeSeries` with columns named in its
  description (G-032). Second batch: 000044's default call lists its
  linearized series as a second Position candidate, and the container name is
  refused again (G-012); 000447 stores x, y and speed in one series under
  "centimeters; centimeters/second", which the reader rightly refuses but
  cannot select columns from (G-042). Third batch: position and head
  orientation as one series per axis with a blank unit (G-050); a
  `conversion` of 0.01 on values already in cm, which the reader applies as
  the NWB spec says, giving a 1.1 cm track (G-052); bare `SpatialSeries`
  labelled metres that hold pixels (G-062); a units table whose ids repeat
  across two probes, which `read_units` refuses (G-057); pose with repeated
  first timestamps, which `read_pose` rightly refuses (G-056). Ten datasets.
- **Change.** Search processing modules in `read_intervals`; accept a container
  name when it holds one series; accept an explicit path to any `SpatialSeries`
  or `TimeSeries` (with column choice) in the position and head-direction
  readers.
- **Module.** `io/nwb`. **Changes results.** No.

### 7. Decoding head direction (G-004)

- **Problem.** There is no public heading decoder.
- **Evidence.** Mouse32: pynapple's `decode_bayes` on the 31 ADn units reaches
  8.9° median error (outside tracking gaps), reproducing its tutorial's 19,590
  samples. A reference result neurospatial cannot produce. One dataset is
  enough here: the gap is a missing entry point, not a property of the data.
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
- **Second batch.** 000044's 1 m circular track: its linearized coordinate
  (0–290 cm) wraps 123 times; replay can only decode it as a line (G-038).
- **Change.** Decided: `Environment.from_loop` (decision 6).
- **Module.** `environment`, `behavior`. **Changes results.** No.

### 9. Environments from the stated apparatus (G-110–G-112, G-131, G-132, G-134, G-135, G-150–G-152, G-155)

- **Problem.** Building the stated apparatus is the faithful path, and it
  is not yet the easy one. Wrong results:
  - `Environment.from_graph` drops graph edges missing from `edge_order`
    but still counts positions on them as on the track (G-112).
  - A sample outside a track's width splits `detect_laps`' recording, so a
    narrow track loses most laps (G-151).
  - `detect_laps` reports a lap on the lattice's 3-D bar graph (G-134).
  - Border scores are -1 for most cells where the tracked head cannot
    reach the wall (G-131).
  - Fewer finite grid scores on a declared arena with unvisited bins
    (G-135, cause not traced).

  Silent hazards:
  - Occupancy and rate maps drop positions outside a declared track
    without a warning (G-155).
  - A wrong length unit inside a declared environment collapses the
    positions into a few bins (G-132).
- **Evidence.** ENVIRONMENT_FIDELITY.md and the two BATCH_FAITHFUL
  documents, on the 30 declared recordings; each gap entry gives the
  counts.
- **Change.**
  - Separate the apparatus's physical width from the placement tolerance
    (G-152; J16 is the case).
  - Bridge laps over samples outside the track, and refuse 3-D graphs in
    `detect_laps`.
  - Fix or refuse `edge_order` gaps in `from_graph`.
  - Add a border score that can reach a wall the tracked point cannot, as
    BNT's inward lookup does, and warn when `observed_wall_fraction` is
    small.
  - Warn on dropped positions and on tiny coverage of a declared
    environment.
  - Docs:
    - `from_samples` cannot see walls thinner than a bin (G-110).
    - Add a recipe for climbing structures (G-111).
    - Correct the widthless-track docstrings (G-150).
- **Module.** `environment`, `behavior`, `encoding`. **Changes results.**
  Yes, for laps on declared tracks and for border scores.

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
  called replay 7/8 times with sweeps and 0/16 alone (simulated events, 100
  shuffles per null). Stationary events can be replay; the point is the batch
  dependence. Say it in the recipe and the replay guide.

## Performance

Nothing to fix at these sizes. Gardner's 483 units: rate maps 4.4 s, decoding
2.9 s, 1.1 GB traced peak. 000044 is read by HTTP byte range from 9 GB files in
about a minute per loader. The slowest operation is the replay comparison on
200 candidate events with published shuffle counts (511 s on the circular
track, 387 s on the linear). The full driver on fourteen recordings with all
112 probes takes 35 min and peaks at 8.5 GB resident. One real problem: with
the default `diffusion_kde`, a rate map on a 98,000-bin grid (pixels taken
as metres) passed 7 GB for one unit, while `binned` took 1 s; the only signal
is the environment's memory warning (G-061).

## Reference-check verdicts

| Check | Verdict |
| --- | --- |
| 1. Spatial information vs pynapple (000582) | Agree by convention: within 2.1 %, all from the spike-to-sample rule (G-003). The default `binned` map is now unsmoothed (G-002, fixed) |
| 2. Grid score vs opexebo (000582); 2b–2c on Gardner and Vollan populations, with the authors' scores | Disagree because of the overlap rule; option (a) agrees best with opexebo and the authors (proposal 3) |
| 3. Head-direction decoding vs pynapple (Mouse32) | neurospatial cannot do it (proposal 7); pynapple's 8.9° is the target |

## Data quirks: detect, or leave to the loader

| Gap | Quirk | neurospatial should |
| --- | --- | --- |
| G-030 | Radians labelled degrees | Detect (proposal 1) |
| G-026 | Linearized loop wraps around | Support it (proposal 8) |
| G-028 | Behavior starts before the neural recording | Leave to `spike_window=` when the recording span is known; nothing warned about the bat's 1.35 s of flight before the first spike |
| G-025 | Positions outside the stated room bounds | Leave to the loader; only an environment with declared bounds could flag them |
| G-008 | Head-direction frame not stated | Keep it visible: `read_head_direction` returns `frame=None` for the file's empty `reference_frame`; say in its docs that None means the file does not state where 0 points |
| G-033 | Labels disagree with values (mm, period as rate, "uv") | Leave to the loader: these are generic `TimeSeries` the readers do not read |
| G-009, G-034, G-035 | One LED; ΔF/F scale; inverted LFP conversion | Nothing |
| G-040 | Speed-filtered time series; sessions table shorter than the data | Leave to the loader; the masked-data warning already flags it |
| G-041 | No ripple table (000044) | Leave to the loader (population-burst candidates) |
| G-053 | Lost-LED code (0, 255); position sampled only at spike times | Leave to the loader; a corner code other than the origin is not detectable by check 1 |
| G-158 | Runs of identical positions (001880 29 % of frames, 000568, 000987): lost tracking or stillness | Leave to the loader, which keeps the positions and warns with the counts; nothing in these files tells lost tracking from stillness |
| G-054 | Video clock 23 s off the position clock | Leave to the loader; undetectable from the arrays |
| G-055 | Repeated timestamps; spikes cover 270 of 1,504 s tracked; spike-clock alignment assumed | Refused / warned already; leave the alignment to the loader |
| G-051 | Motion-capture heading 0.3–0.5 rad off the direction of movement, varying between windows | Leave to the loader; the frame check needs a decision (below) |
| G-019, G-036, G-039, G-059 | Pre-registered expectations that were wrong | Nothing (kept as registered) |

## Should loader code move into neurospatial?

No, except through the readers. The fourteen loaders total about 2,400 lines
(the first eight about 870); of that, roughly 15–20 lines each are the pinned download and the
standalone summary, and the rest is specific to one file: column names, unit
fixes, epoch selection, a restricted unpickler for one pickled dataset, a
MATLAB-table reader for one zip. The generic parts that matter to users are
the NWB reader gaps in proposal 6 and the unit checks in proposal 1, which
belong in `io/nwb`. A `neurospatial.datasets` fetcher (pooch-style) is only
worth it if tutorials come to depend on three or more public recordings; the
harness's `_fetch.py` (77 lines) would be the starting point.

## Next batch

No new recordings are planned for the survey. Next, in
`.claude/docs/plans/real-data-survey-batches/` and
`.claude/docs/plans/dataset-loaders/`, comes a check of each operation
against an independent truth: the file's own values, simulations on real
behavior and environments, published results, or naive numpy. That work
also covers messiness, usability and visualization, and public
`neurospatial.datasets` loaders for the best-known recordings.

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
| G-001 | Fixed (#265) |
| G-010 | Warning calibration, check 5 (regressed for `epochs`; fix pending) |
| G-002 | Fixed (#262) |
| G-003 | Docs |
| G-004 | Proposal 7 |
| G-005, G-016 | Friction |
| G-006 | Proposal 3 |
| G-043 | Resolved: a harness bug in the shift null (proposal 4) |
| G-007, G-023 | Docs |
| G-008, G-009, G-019, G-025, G-028, G-033–G-036, G-039–G-041 | Data quirks |
| G-011, G-012, G-032, G-042, G-050, G-052, G-056, G-057, G-062 | Proposal 6 |
| G-013 | Fixed (#262) |
| G-014 | Warning calibration, checks 2 and 4 |
| G-058 | Friction: `units="pixel"` warns; only "pixels"/"px" are registered |
| G-060 | Missing: no 2-D replay score for an open maze |
| G-061 | Performance |
| G-090 | Data quirk: hc-11 and 000044 differ only in sleep scoring |
| G-091, G-092, G-093 | Data quirks (CRCNS): wrong expectations; hc-28 heading offset; hc-31 inches and µs |
| G-094 | Warning calibration: coincident LEDs read as heading 0 |
| G-095 | Duplicate of G-058 |
| G-096 | Warning calibration: −1 untracked marker kept, undetected |
| G-097 | Warning calibration: spike clock assumed 20 kHz, silent with epochs |
| G-098 | Friction: `heading_vs_movement` is biased on whole-pixel positions at 30 Hz (a `lag=` option added) |
| G-070, G-071, G-073–G-077 | Data quirks (fourth batch) |
| G-072 | New questions: frame check |
| G-078 | Fixed in the harness: decoding chance is geodesic and shuffled within connected parts |
| G-079 | Friction: the large-grid warning suggests a bin smaller than the one in use |
| G-080 | Warning calibration: pixels taken for cm, undetected |
| G-081 | Warning calibration: header clock 0.17 % off, undetected |
| G-082 | Missing: no boundary-vector/egocentric-boundary verdict; no `min_speed=` there |
| G-015, G-037 | Proposal 2; G-015 partly fixed (#265), open arenas remain |
| G-017 | Warning calibration, check 1 |
| G-018 | Dropped (decision 4) |
| G-020 | Docs; undetectable |
| G-021, G-026, G-038 | Proposal 8 |
| G-022 | Proposal 4 |
| G-024 | Won't fix (behaves as documented) |
| G-027 | Docs |
| G-029 | Proposal 5 |
| G-030, G-031 | Proposal 1 |
| G-110–G-112, G-131, G-132, G-134, G-135, G-150–G-152, G-155 | Proposal 9 |
| G-113, G-115 | Harness: one environment per apparatus; the graph environment where a loader gives one |
| G-114, G-116, G-117, G-130, G-153, G-154 | Data quirks (declared environments) |
| G-133 | Friction: egocentric boundary maps in pixels need `distance_range` |
| G-156, G-157 | Harness: fidelity study limits (`jump_links` at a hub; support reported, not judged) |
| G-158 | Data quirk: runs of identical positions (lost tracking or stillness) are warned about, not masked |
| G-159 | Data quirk (unconfirmed): the lattice's head samples may sit 9 cm horizontally off the authors' bars; not verified |
| G-160 | Harness: a track's offset is read in a window narrower than 000044 linear's lateral scatter; both estimators read, not verified when they straddle |

## Decisions for the maintainer

Answered 2026-11-02.

1. **Head-direction label check (proposal 1).** Yes, as a warning:
   `read_head_direction` warns when a series labelled degrees has values that
   look like radians, naming both readings. The coverage warning (check 3)
   waits for the linear-track calibration.
2. **Grid score (proposal 3).** Wait for more grid cells (Gardner 2022) before
   choosing; no change until then.
3. **Spatial information bias (proposal 4).** Run the shuffle-null test of
   `spatial_information_bias` on sparse occupancy first. The warning plan
   (warn when most visited bins hold very little time) is agreed in principle
   and lands only if the test shows the bias estimate misses the effect.
4. **Warnings.** Check 2 (fewer than 5 bins along a dimension): yes. Check 1
   (samples exactly at the origin): yes, after more recordings confirm it,
   including the start-of-track case. The rule for replay events too short to
   score: no.
5. **Laps (proposal 2), decided 2026-11-03.** `detect_laps`'s template
   methods go: `method="auto"` (first circuit as template, 80 % overlap)
   finds 1 of 97 laps on the 000044 circular track and 1 of about 107 on the
   hc-18 train, and gives false laps in open arenas and on a W-track;
   `method="reference"` uses the same matching. A start region drawn across
   the track (`method="region"`) finds 96–97 of 97. So: remove `auto` and
   `reference`; require `start_region=` except on loop environments
   (`Environment.from_loop` or a cyclic track graph), where a lap is one full
   wrap of progress along the loop; and stop silently dropping laps that cross
   a tracking gap longer than `max_gap` (86 of 97 with the default 1 s).
   Laps in open arenas exist only if the user draws a start region.
6. **New capabilities.** Plan NWB reader coverage (6) and periodic 1-D tracks
   (8). Loop tracks get a factory, `Environment.from_loop(length=, bin_size=,
   track_width=)`, that builds a cyclic track graph laid out on a circle of
   that circumference, plus a helper that maps linear position onto it; a
   cyclic track graph already wraps distances and laps. Defer
   continuous-signal rate maps (5) and heading decoding (7); heading decoding
   is wanted later.
7. **Next batch.** Run more datasets, chosen by coverage, with the long-term
   aim of loading most of the catalog. The restricted unpickler approach
   applies to pickled formats such as Gardner 2022.
8. **Tutorial.** Deferred.

### Evidence for the decisions that waited on data (second batch)

- **Decision 1, heading coverage.** Calibrated on a linear track: 36/36 bins
  (proposal 1). The check can land with the reader warning.
- **Decision 2, grid score.** Recommend option (a), a NaN-aware Pearson at
  every lag with at least about 20 overlapping visited bins (proposal 3): it
  fixes the large-spacing collapse (Vollan module 3, 93 cm spacing: 0.16
  today, 0.77 with (a), 0.68 by the authors), stays within 0.02 of the
  authors on modules 1–2, and agrees best with opexebo on 483 Gardner units.
  On Vollan, only zero-filling matches opexebo closely.
- **Decision 3, information bias.** The estimate holds on sparse occupancy;
  the gap is burstiness, not sparseness (proposal 4). Recommend docs that call
  it a Poisson null and point to shuffle verdicts; no sparse-occupancy
  warning.
- **Decision 4, check 1.** Restrict it to 2-D/3-D positions: the start of a
  linearized track is a real place, and the circular track's file puts 0.23 %
  of samples exactly there. With that restriction it fires on no clean
  recording and catches all 18 silent missing-as-zero probes.
- **Decision 5, laps.** Decided (above): the template methods go; laps need a
  start region except on loop environments.

### New questions from the third batch

- **Check 2 needs a revised rule.** The accepted "fewer than 5 bins along a
  dimension" fires on a clean narrow track (001695, 4 bins across);
  "fewer than 5 along every dimension" keeps every ÷100 catch.
- **Heading-frame check for motion capture (G-051).** Both motion-capture
  recordings sit 0.3–0.5 rad off the direction of movement on average, and
  the offset varies between 60 s windows (Kanter: −0.62 to +2.33 rad), so it
  is not one constant to subtract. The loaders keep the headings marked
  "frame not verified" and refuse only when the spread exceeds 45–70°
  (negating the heading raises the spread to 81–87°). J16's orientation sits
  a constant +13° off (spread 20°) and is dropped by the 0.2 rad rule.
  Should the check separate frame errors (about 90° or 180°) from small
  offsets, and keep or drop such headings?
- **Boundary-vector and egocentric boundary cells (G-082).** The new
  `rates.egocentric_boundary` operation works on simulated and real data,
  but there is no shuffle verdict for these cells and no `min_speed=` on the
  boundary-vector entry points. A candidate proposal.

