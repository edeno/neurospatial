# Faithful environments: tracks and mazes

Twenty-two track and maze recordings were given the apparatus their sources
state, as a track graph, in place of a grid built from the samples. 19 are
declared; 3 fitted layouts were refused on misfit evidence and stay
inferred. This page lists, per recording, what was declared and why, the
source, how the layout was placed and how well it fits, and what changed
between the run on the declared environment and the run on the samples' grid
(`run.py --inferred-env`). The evidence is in [GAPS.md](GAPS.md) (G-150 to
G-156), [results/](results/) (declared), [results/inferred_env/](results/inferred_env/)
(inferred), `results/studies/declared_vs_inferred.md` (every value side by
side) and `results/studies/environment_fidelity.md` (the fidelity study of
both environments). [FINDINGS.md](FINDINGS.md) is not edited here.

## How a track is declared

A loader sets `geometry["environment"]`, a zero-argument callable returning
the environment, built in `load()` from the clean samples
(`datasets/_tracks.py`, `declared_geometry`):

- **Topology and stated sizes from a source; placement from the samples or
  the file.** Where the file holds the authors' graph or centreline
  (labyrinth, J16, Takigawa, Hafting's normalised frame) or its own
  coordinate (hc-31, and 000044's linear track placed along its linearized
  coordinate), it is used as stored. A straight track of stated length lies
  along the samples' principal axis, centred between their 0.5th and 99.5th
  percentiles. A maze whose layout is stated (a W, a radial maze, a Z, an X,
  a figure-8, a 3-arm maze) starts from candidate layouts drawn on the
  samples and has its node positions fitted (a truncated squared distance
  from the samples to the graph plus from points along each edge to the
  nearest sample, L-BFGS-B on a seeded 6,000-sample subsample); the radial
  maze's arms are then refitted to their own samples. The circle is the
  stated 1 m diameter placed by a robust circle fit; the train circuit,
  whose shape no source states, is the samples' median radius in 72
  sectors.
- **Every edge kept.** `Environment.from_graph` drops edges missing from
  `edge_order` (G-112); `_geometry.graph_environment` lists all of them, so
  the figure-8, the circle and the circuit are cyclic graphs.
- **Physical width apart from the placement tolerance (G-152).** In
  `Environment.from_graph`, `track_width` only decides which positions are
  placed and the `TrackArena` footprint; the bins, their lengths, their
  links, kernels and the occupancy and rates of placed samples do not
  depend on it (read in `src/`, checked at widths 6 and 30 cm). The tracked
  point is usually an LED on a headstage above an unwalled track, which
  swings past its edge, so each track is built with a placement tolerance
  measured from the samples (twice their 99th percentile distance from the
  track, at least the physical width and a bin; 000044's linear track the
  95th, because its first 110 s are an excursion off the track), about 1 %
  of samples outside by construction. The physical width, stated by a source
  or chosen by the loader (the labyrinth's one cell), is recorded apart
  (`geometry["declared_physical_width"]`) with the share of samples beyond
  half of it, split by speed: a tracking measure, not a verdict. What was
  tracked is recorded in `geometry["tracked_point"]` ("not stated" where
  the source does not say).
- **Thresholds set after seeing the data.** The 20-bin stretch was 10 bins
  until the 000044 circle was refused on one short side under its reward
  area (commit 4b865baca); it was then widened. The stretch, bias and
  support limits are in bins, so in physical units they vary with each
  recording's bin, from 0.5 cell (the labyrinth) and 2-4 cm to 5 px
  (hc-18). How sensitive these limits are to misfits of known size is not
  measured here; it is the survey's simulation ladder (phase 6, rungs 2-3,
  planted misfits).
- **Refusal on misfit, not scatter.** A layout fitted to the samples is
  refused when a straight stretch of track (at least 20 bins, not crossing
  a corner) has its samples a median of more than one bin to one side, when
  a quarter of a segment has no sample within a bin, or when a stated
  length is more than 15 % off (`_geometry.MAX_SIZE_RESIDUAL`: one
  tolerance for every stated size, shared with the arena loaders, for
  "about" in a stated size and a head leaning past a track's end; not
  tuned). The authors' geometry and the files' own coordinates are not
  fitted, so they are not refused; the fidelity study judges them with the
  same evidence (`_geometry.judge_declared`, one function for any declared
  environment), and also how far the samples reach past a track's free
  ends (summed over the ends, at most 15 % of the track's length: a track
  drawn too short, which the lateral and support measures do not see) and,
  on an outline, how far they fall short of its side (at most 15 %: the
  samples may fall short of the walls but not reach past them).

The driver uses a declared track as the graph environment too (laps, turns),
decodes replay on it (status `ok` where it was a G-038 workaround on the
file's linearized coordinate), counts laps a second way, by a start region
the loader states (`geometry["lap_start_region"]`, `method="region"` in the
analysis epochs), reports grid, border and egocentric-boundary scores as
n/a on a track, and gives the decoding error over the time bins whose true
position the declared track places in both runs (`median_error_on_declared`).

## Per recording

Values are declared / inferred. Physical width: stated, chosen, or not
stated. Beyond: the share of tracked samples farther than half the physical
width from the track, then the same share stopped / moving (up to 30 cm/s)
/ fast (the labyrinth: below / above 1 cell/s). Tolerance: the
`track_width` the track is built with and, in brackets, the share of all
tracked samples it leaves out (hc-18's 92.8 % are its sleep-box sessions,
off the circuit; 1 % of its travel samples). Lateral: the samples' median
|lateral offset| and the worst stretch's median signed offset, in the
file's units, on the analysed samples. Decoding error: median over the same
time bins in both runs (those the declared track places), along each run's
own environment (the track graph, or the samples' grid); pixels for hc-3
and hc-18. Laps: default, by region. Replay: the `GROSMARK_2016` detection
rate (`wk`: the G-038 workaround on the linearized coordinate).

| Recording | Status | Layout (source) | Physical width | Beyond | Tolerance (outside) | Lateral | Occupancy (s) | Info (bits/spike) | Decoding error | Laps | Replay | Fidelity verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rosenberg_labyrinth | declared | binary-tree labyrinth, the repository's NewMaze graph | 1 cell chosen | 4.6 % (11 / 1) | 1.26 (1.0%) | 0.16, 0.34 | 4797 / 4808 | – / – | – / – | – | – / – | mismatch |
| j16_23020757 | declared | the authors' graph, 10 nodes, 9 corridors (file) | 6.5 cm stated | 63.8 % (87 / 64 / 35) | 36.9 (1.0%) | 2.73, 1.81 | 792 / 802 | 2.05 / 2.58 | 2.0 / 5.7 | – | – / – | matches |
| takigawa_7114733 | declared | two tracks, the authors' centrelines (file); Tirole et al. 2022: 2 m | not stated | – | 41.4 (1.0%) | 4.57, 4.12 | 1069 / 1093 | 1.66 / 2.22 | 7.7 / 13.7 | – | – / – | mismatch |
| hafting_2008_nird | declared | straight 320 cm (archive readme), the file's frame | not stated | – | 31 (1.0%) | 3.21, 1.24 | 650 / 656 | 0.98 / 1.05 | 104.0 / 157.0 | – | – / – | matches |
| grosmark_000044_linear | declared | 15 cm reward area + 1.6 m stem + 15 cm reward area (Grosmark & Buzsaki 2016), on the file's linearization | not stated | – | 55 (5.0%) | 9.04, 1.53 | 981 / 1052 | 0.38 / 0.81 | 2.5 / 8.5 | – | 0.155 / 0.140 (wk) | mismatch |
| vollan_lt | declared | straight 200 cm (Vollan et al. 2025) | not stated | – | 31.3 (1.0%) | 3.19, 0.21 | 316 / 319 | 0.15 / 0.29 | 5.0 / 8.5 | – | – / – | matches |
| valero_000568 | declared | straight 110 cm (NWB description) | 6.35 cm stated | 12.4 % (12 / 14 / 9) | 12.9 (1.0%) | 1.22, 0.07 | 944 / 950 | 0.28 / 0.29 | 34.2 / 32.1 | – | 0.000 / error | matches |
| aeryjones_000165 | declared | straight 36 cm (Aery Jones et al. 2021) | not stated | – | 5.42 (0.9%) | 0.39, 0.15 | 270 / 270 | 0.01 / 0.01 | 16.0 / 17.0 | – | – / – | matches |
| gonzalez_001695 | declared | straight, length fitted (Gonzalez et al. 2026: wells at either end) | not stated | – | 3.16 (1.0%) | 0.42, 0.02 | 545 / 549 | 0.22 / 0.22 | 17.2 / 18.5 | – | 0.040 / error | not verified |
| hc3_ec013_669 | declared | straight 250 cm (Mizuseki et al. 2012), pixels | 8.04 px stated (7 cm) | 26.3 % (19 / 34 / 15) | 29.3 (1.0%) | 2.13, 0.03 | 530 / 535 | 0.57 / 0.67 | 8.5 / 12.2 | – | – / – | not verified |
| hc31_anm190422 | declared | 1-D, the authors' 48 m coordinate (hc-31) | not stated | – | 2.5 (0.0%) | 0.00, 0.00 | 909 / 909 | 3.26 / 3.25 | 574.8 / 572.5 | – | – / – | matches |
| figure8_000552 | declared | figure-8, both loops closed (DANDI 000552), fitted | not stated | – | 12.5 (1.0%) | 0.99, 0.60 | 597 / 603 | 0.39 / 0.40 | 7.8 / 8.8 | 6, 14 / 0, 20 | 0.014 / 0.028 (wk) | not verified |
| grosmark_000044_circular | declared | 1 m circle, 72-gon loop (Grosmark & Buzsaki 2016) | not stated | – | 33.9 (1.0%) | 3.05, 2.38 | 1709 / 1726 | 0.66 / 0.83 | 6.5 / 11.0 | 67, 65 / 1, 85 | 0.055 / 0.095 (wk) | matches |
| hc18_train261 | declared | closed circuit from the travel samples (no shape stated) | not stated | – | 35.2 (92.8%) | 2.84, 1.20 | 1423 / 1443 | 0.18 / 0.19 | 276.1 / 303.5 | 35, 31 / 1, 94 | – / – | not verified |
| jadhav_000447 | inferred (refused: base +17% of the stated length; arms +19% of the stated length) | W-track (Tang et al. 2023: ~80 x 80 cm), fitted | 7 cm stated | 35.8 % (22 / 50 / 44) | – | – | 912 / 912 | 0.58 / 0.58 | 18.5 / 18.5 | – | – / – | – |
| jadhav_000447_familiar | declared | W-track (Tang et al. 2023: ~80 x 80 cm), fitted | 7 cm stated | 13.0 % (2 / 32 / 26) | 29.7 (1.0%) | 0.79, 0.78 | 322 / 331 | 1.00 / 1.08 | 35.9 / 40.6 | – | – / – | matches |
| hc28_hpa | declared | W-track (Jadhav et al. 2016: ~80 x 80 cm), fitted | 7 cm stated | 37.3 % (24 / 38 / 57) | 20.8 (1.0%) | 2.68, 1.00 | 800 / 811 | 0.85 / 0.92 | 17.1 / 16.6 | – | – / – | matches |
| agarwal_24757638 | declared | 3-arm maze (Figshare; Agarwal et al. 2026), fitted | not stated | – | 16.8 (1.0%) | 0.70, 0.80 | 1028 / 1053 | 0.18 / 0.19 | 3.9 / 5.7 | – | – / – | not verified |
| shipley_30902972 | declared | 8-arm radial maze (Shipley et al. 2026: 80 cm arms, 24.7 cm platform), fitted | 10 cm stated | 0.7 % (1 / 0 / 1) | 10 (0.3%) | 1.06, 0.97 | 1344 / 1350 | 0.24 / 0.24 | 62.3 / 52.2 | – | – / – | matches |
| olafsdottir_5566548 | inferred (refused: last section -16% of the stated length) | Z-track (Olafsdottir et al. 2016: 190/220/190 cm), fitted | 10 cm stated | 7.6 % (30 / 9 / 7) | – | – | 1266 / 1266 | 0.84 / 0.84 | 115.3 / 115.3 | – | – / – | – |
| giocomo_001701 | declared | X maze (Aery Jones et al. 2026), fitted | not stated | – | 5.68 (1.0%) | 0.92, 0.38 | 956 / 967 | 0.12 / 0.13 | 2.5 / 3.5 | – | – / – | not verified |
| dyads_001845 | inferred (refused: 80% of a segment has no sample within a bin (limit 25%)) | W-maze (Shukla et al. 2026: 80 x 80 cm), fitted | 7 cm stated | 6.1 % (0 / 16 / 16) | – | – | 31 / 31 | – / – | – / – | – | – / – | – |

### Choices that changed a recording

- **000447 split in two.** The novel and familiar W-tracks lie rotated 90
  degrees in one camera frame and their arms cross, so one environment with
  two components would put each maze's samples on the other's arms where
  they overlap. `jadhav_000447` is now the novel W-track (runs 1 and 3;
  positions outside them are NaN, its own trajectories, `analysis_epochs`
  its runs), and `jadhav_000447_familiar` is run 2, a new registry entry
  sharing the loader (`load_jadhav_000447.load_maze`). Tang, Shin & Jadhav
  2023 give one size for both; the familiar maze fits it (base -1 %, arms
  +6 %) and the novel one does not (base +17 %, arms +19 %; G-153), so the
  novel recording stays inferred.
- **The figure-8's positions are the camera (x, y).** A loop cannot be
  declared on the linearized line, where 0 and 222 cm are different points
  (G-021); the linearized coordinate moved to `params["linearized"]`, as on
  the other 2-D tracks, and the two range expectations now check it. The
  inferred run is therefore the camera grid (667 bins), not the old 1-D
  environment.
- **000165 is in cm.** Aery Jones et al. 2021 (Cell Reports 37:110159,
  read in the PMC9069800 full text) state "a 36 cm linear track"; the
  tracked span is 35.4 cm along the axis, so the file's cm label is right
  and the loader no longer returns arbitrary units.
- **000044 linear: reward areas as end segments.** The stated 1.9 m maze is
  a 1.6 m stem with a 15 cm reward area continuing it at each end, placed
  along the line the file's linearized coordinate traces (0.963 cm of 2-D
  distance per linearized cm). The reward-end samples spread up to about
  27 cm across the axis, so the reward segments' centreline is empty over
  44 % of its length: the study calls this misfit, and the layout has no
  width or shape for the reward areas to do better.
- **The labyrinth at one chosen cell.** The paper's corridors are 1-1/8 in
  on a 1.5 in grid, 0.75 of a cell; one cell, the corridor pitch, is chosen
  so a corridor ends at the wall it shares with its neighbour. 4.6 % of nose
  samples lie beyond half a cell (17.9 % beyond half the paper's corridor).
- **Shipley's centre.** The radial maze is centred on the middle of the
  samples' extent: the densest point is a reward well, and starting there
  put three fitted arms on one arm (the first run of this batch reported
  8.8 % outside for that wrong layout; centred, 0.7 % of maze samples are
  beyond the 10 cm arms).

## What changed between declared and inferred

- **Connectivity is the apparatus's.** Every declared track is one part
  (Takigawa two, as stated). Against its own geometry a declared track
  measures only its binning (the labyrinth's path along its bins 0.95 x the
  distance along the graph it was built from, Takigawa's 0.97), so these
  are not evidence of fit; what they replace is the samples' grid, whose
  path against the same references is 0.66 and 0.87 and which links
  corridors through walls (`jump_links` in the fidelity study).
- **Samples placed.** A declared track places 99 % of tracked samples (95 %
  on 000044's linear track) by construction: its placement tolerance is
  twice the samples' 99th percentile (95th) distance from it, so this is
  not evidence of fit. Occupancy is
  within 3 % of the grid's on every declared recording but that one (981
  against 1,052 s).
- **Decoding error over the same time bins**, median, declared then
  inferred. The two are different metrics: the declared error is the
  distance along the track graph, the inferred one the distance along the
  samples' grid bins, which can cut across a maze where the graph cannot,
  so the pairs are reported and not ranked. 000044 linear 2.5 (graph) and
  8.5 cm (grid), the circle 6.5 and 11.0, Takigawa 7.7 and 13.7, Vollan 5.0
  and 8.5, J16 2.0 and 5.7, hc-3 8.5 and 12.2 px, Hafting 104 and 157 cm,
  Agarwal 3.9 and 5.7, the X maze 2.5 and 3.5, 000447's familiar maze 35.9
  and 40.6, Shipley's maze 62.3 and 52.2, hc-28 17.1 and 16.6, 000568 34.2
  and 32.1.
- **Laps exist on the loops.** `detect_laps`' default, which found 1 lap on
  the samples' grids of 000044's circle and hc-18's circuit (G-037), finds
  67 and 35 on the declared loops, and 6 on the figure-8 (0 on the grid). By
  start region the declared loops give 65, 31 and 14, the samples' grids
  85, 94 and 20 (truth: about 97 clockwise laps, about 107, and 21 trials).
  The declared counts are lower because the 1 % of samples outside the
  tolerance each split the recording (G-151): with no sample outside, the
  figure-8 gives 20 and hc-18 92-97.
- **Replay runs on the track.** 000568 and 001695, which errored (G-038),
  decode on the declared track (Grosmark rates 0 and 0.040); 000044 and the
  figure-8 no longer need the file's coordinate (`ok` instead of a
  workaround): 000044 linear 0.140 -> 0.155, the circle 0.095 -> 0.055,
  the figure-8 0.028 -> 0.014. Davidson rates are 0 everywhere in both
  runs.
- **Spatial information is lower on most tracks** (000044 linear 0.81 ->
  0.38 bits/spike, Vollan 0.29 -> 0.15, Takigawa 2.22 -> 1.66, J16 2.58 ->
  2.05, the circle 0.83 -> 0.66) and about the same on the mazes. A track
  bin pools positions across the track that the 2-D grid keeps apart, so
  the grid's extra information is partly lateral position; that is a
  hypothesis, not tested here.
- **Area-only scores are n/a on a track** (grid, border, egocentric
  boundary). On the samples' grids the border score refused on 19 of the 22
  (G-005) and the grid score had a finite value on 5.

## Head scatter past the physical width

On every track with a stated width, a share of head samples is beyond it
(Beyond, above), and on none is it misplacement: the worst stretch's median
offset is at most 1.81 cm (J16). On J16 the authors' graph is placed right
(median signed offset -1.3 to +1.8 cm per corridor), only 13.1 % of samples
are more than 10 cm off, and neither a uniform scale (1.024, 63.7 % ->
57.2 % outside) nor a translation (50.4 %) explains it; the share falls
from 87 % stopped to 35 % above 30 cm/s, and while running it is 61 % in
the junction-end tenth of each edge against 34 % mid-edge (G-152). On the
W-tracks and hc-3 it is highest while moving (hc-28 24 / 38 / 57 %), on J16
and the Z-track while stopped: swings at turns and posture at the wells.
This is the case for a placement tolerance separate from the physical width
in neurospatial (G-152).

## Gaps

- G-150: `from_graph`, `linear_track` and `maze` document nearest-track
  placement without a width; `bin_at` refuses it (docs).
- G-151: one sample outside a track's width splits `detect_laps`'
  recording; at 1-5 % outside most laps are lost (figure-8, hc-18).
- G-152: one `track_width` is both the physical width and the placement
  tolerance; head tracking leaves the physical width (J16 63.8 %, hc-28
  37.3 %, hc-3 26.3 %, with the speed split and the J16 analysis).
- G-153: fitted mazes against their stated sizes: 000447's novel W-track
  (+17 / +19 %) and the Z-track (last section -16 %) are refused, 001845 for
  its unsampled base; hc-28 -11 %, the familiar W-track +6 %, Shipley -6 to
  +3 %.
- G-154: 000044's circular coordinate leaves out the 30 cm reward area,
  which explains G-039's 72 % "on track" and its 134 cm extent.
- G-155: occupancy and rate maps drop positions outside a declared track
  without a warning; `detect_laps` and `decoding_error` do warn.
- G-156: the fidelity study's `jump_links` flags links inside an open hub,
  where the arms' corridors overlap (a synthetic 8-arm star: 128 of 3,504
  links, all within 17.4 cm of the centre); no judged reference here has
  one.

Existing gaps touched: G-110 and G-113 have a harness answer now (the
labyrinth's graph, 000447 split), and G-110's links through the walls are
found by the layout (`jump_links`), not by sample gaps, which cannot see a
wall thinner than a bin; G-021 and G-038 for the figure-8 and 000044
(declared loops, replay on the track); G-037 (defaults now find laps on a
declared loop, still too few); G-016 (J16's and Hafting's graphs now have
a tolerance).

## What could not be verified or done

- Widths: no source read states the width of 000044's tracks, Vollan's
  linear track, 000165, 001695, Hafting's track, Takigawa's tracks, the
  figure-8, Agarwal's maze or the X maze; those tracks have no physical
  width recorded, only the measured tolerance. hc-31's layout (Rich et al.
  2014 not readable) and hc-18's circuit (Drieu et al. 2018's supplement not
  readable) are not stated.
- What Agarwal's "2.13m-long" and the X maze's "130-cm" span is not stated,
  so neither length is compared; 001695's size is in a supplementary table
  that could not be read.
- hc-3's length is not an independent check: its pixel scale comes from the
  same 250 cm.
- Fidelity verdicts of the 19 declared tracks: 10 match, 3 mismatch, 6
  not verified (ENVIRONMENT_FIDELITY.md). A match rests on a stated size
  against the samples' own or on the authors' geometry as stated; the
  references built from the samples (principal axis, linearized
  centreline) judge nothing. The labyrinth's mismatch is its graph ending
  at dead-end cell centres (samples reach 15.6 % of its length past its 65
  ends), not a scale error. Hafting's and hc-31's lengths (+0.0 %, +0.3 %)
  confirm the files' frames, not the tracks. The circle's worst stretch is
  2.38 cm to one side against a 2.5 cm limit. How sensitive these limits
  are to misfits of known size is the survey's phase 6 (simulation ladder
  rungs 2-3), not measured here.
- The Z-track (-16 %) is still refused and hc-28 (-10.5 % / -11.1 %) still
  declared under the one 15 % size tolerance; no verdict changed with it.
- The tracked point is "not stated" for 13 recordings; where it is
  recorded, several strings rest on the loader's reading of its source
  ("per the loader").
- Lap counts by region depend on the tolerance (G-151); the truths quoted
  are the unwrapped angle (G-037) and the file's trial table.
