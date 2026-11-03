# Faithful environments: tracks and mazes

Twenty-two track and maze recordings now hand the driver the apparatus their
sources state, as a track graph, instead of a grid built from the samples.
This page lists, per recording, what was declared and why, the source, how
the layout was placed and how well it fits, and what changed between the
run on the declared environment and the run on the samples' grid
(`run.py --inferred-env`). The evidence is in [GAPS.md](GAPS.md) (G-150 to
G-154), [results/](results/) (declared), [results/inferred_env/](results/inferred_env/)
(inferred), `results/studies/declared_vs_inferred.md` (every value side by
side) and `results/studies/environment_fidelity.md` (the fidelity study of
both environments). [FINDINGS.md](FINDINGS.md) is not edited here.

## How a track is declared

A loader sets `geometry["environment"]`, a zero-argument callable returning
the environment, built in `load()` from the clean samples
(`datasets/_tracks.py`):

- **Topology and stated sizes from a source; placement from the samples.**
  Where the file holds the authors' graph or centreline (labyrinth, J16,
  Takigawa, Hafting's normalised frame) it is used as stored. A straight
  track of stated length lies along the samples' principal axis, centred
  between their 0.5th and 99.5th percentiles, or along the file's own
  linearized coordinate (000044 linear, hc-31). A maze whose layout is
  stated (a W, a radial maze, a Z, an X, a figure-8) starts from candidate
  layouts drawn on the samples' minimum rotated rectangle and has its node
  positions fitted: the mean of a truncated squared distance from the
  samples to the graph plus the distance from points along each edge to the
  nearest sample (so an arm cannot run past the samples), minimised with
  L-BFGS-B on a seeded 6,000-sample subsample. The circle is the stated 1 m
  diameter placed by a robust circle fit; the train circuit, whose shape no
  source states, is the samples' median radius in 72 angular sectors.
- **Every edge kept.** `Environment.from_graph` drops edges missing from
  `edge_order` (G-112); `_geometry.graph_environment` lists all of them, so
  the figure-8, the circle and the circuit are cyclic graphs.
- **Width.** The stated width where a source gives one. Otherwise it is
  measured: twice the samples' 99th percentile distance from the graph (at
  least one bin), because neurospatial will not place positions on a track
  without a width although its docstrings say it will (G-150). The 99th, not
  the 95th, because every sample outside a track splits `detect_laps`'
  recording (G-151).
- **Residual and refusal.** Each declaration writes one `conversions` line:
  the layout, its source, the width (stated or measured), the placement and
  the fit (median and 95th percentile sample distance to the track, the
  track's length, the share of samples within half the width). A fitted
  placement whose median sample distance exceeds half the stated width (or
  two bins without one) is refused and the recording stays inferred; none
  was refused in the final run. The authors' own graphs and the placement
  by a file's coordinate are not fitted, so they are not refused.

The driver uses a declared track as the graph environment too (laps, turns),
decodes replay on it (status `ok` where it was a G-038 workaround on the
file's linearized coordinate), counts laps a second way, by a start region
the loader states (`geometry["lap_start_region"]`, `method="region"` in the
analysis epochs), and reports grid, border and egocentric-boundary scores as
n/a on a track.

## Per recording

Values are declared / inferred. "Outside": share of tracked samples the
declared environment cannot place (`env.declared`). "Offset": median sample
distance to the declared track, and the share within half its width.
Decoding error is the median along the environment (cm, or pixels for hc-3
and hc-18); on a declared track it is measured only on samples the track
places, so where many are outside (J16, the W-tracks, hc-3) it describes the
samples on the track. Information is the median bits per spike.

| Recording | Declared layout (source) | Width | Placement, fit | Outside | Occupancy (s) | Info | Decoding error (chance) | Laps: default, region | Replay (Grosmark rate) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rosenberg_labyrinth | binary-tree labyrinth, the repository's NewMaze graph (eLife 2021: corridors 1-1/8 in on a 1.5 in grid) | 1 cell, stated (the corridor pitch) | as stored; offset 0.21 cell, 95.4 % | 4.6 % | 4722 / 4808 | – | – | refused (tree) both | – |
| j16_23020757 | the authors' graph, 10 nodes, 9 corridors (file; Comrie et al. 2026: 6.5 cm hallways) | 6.5 cm, stated | as stored; offset 4.23 cm, 36.2 % | 63.8 % | 424 / 802 | 2.37 / 2.58 | 2.0 (108) / 5.7 (113) | refused (tree) both | – |
| takigawa_7114733 | two tracks, the authors' centrelines, two components (file; Tirole et al. 2022: 2 m tracks) | 41.4 cm, measured | as stored; offset 3.70 cm | 1.0 % | 1069 / 1092 | 1.66 / 2.22 | 7.8 (76) / 13.7 (68) | refused both | – |
| hafting_2008_nird | straight track, 320 cm (archive readme) | 31 cm, measured | the file's frame (posx -160 to 160); offset 3.21 cm | 1.0 % | 650 / 656 | 0.98 / 1.05 | 106 (132) / 157 (130) | refused both | – |
| grosmark_000044_linear | 15 cm reward area, 1.6 m stem, 15 cm reward area (Grosmark & Buzsaki 2016) | 55 cm, measured (95th percentile, see below) | the file's linearized coordinate; offset 9.35 cm | 5.0 % | 981 / 1053 | 0.38 / 0.81 | 2.5 (24.5) / 8.5 (34.7) | refused both | 0.155 / 0.140 (workaround G-038) |
| vollan_lt | straight track, 200 cm (Vollan et al. 2025) | 31.3 cm, measured | principal axis, centred; offset 3.97 cm; 11.6 % beyond the ends | 1.0 % | 316 / 319 | 0.15 / 0.29 | 5.0 (62.5) / 8.5 (63.1) | refused both | – |
| valero_000568 | straight track, 110 x 6.35 cm (NWB description) | 6.35 cm, stated | principal axis, centred; offset 1.58 cm, 87.6 %; 26.2 % beyond the ends | 12.4 % | 819 / 950 | 0.31 / 0.29 | 34.2 (39.1) / 32.1 (46.6) | refused both | 0 / error (G-038) |
| aeryjones_000165 | straight track, 36 cm (Aery Jones et al. 2021) | 5.42 cm, measured | principal axis, centred; offset 0.44 cm | 0.9 % | 270 / 270 | 0.009 / 0.010 | 16.0 (7.0) / 17.0 (7.4) | refused both | – |
| gonzalez_001695 | straight track, length fitted (Gonzalez et al. 2026: wells at either end; size not read) | 3.16 cm, measured | principal axis, 105.6 cm; offset 0.42 cm | 1.0 % | 545 / 549 | 0.22 / 0.22 | 17.2 (39.3) / 18.5 (40.0) | refused both | 0.040 / error (G-038) |
| hc3_ec013_669 | straight track, 250 x 7 cm (Mizuseki et al. 2012), in pixels at the inferred scale | 8.04 px, stated | principal axis, centred; offset 2.13 px, 73.7 % | 26.3 % | 379 / 535 | 0.54 / 0.67 | 8.5 (99.5) / 12.2 (114) px | refused both | – |
| hc31_anm190422 | 1-D track along the authors' coordinate, 4816 cm (hc-31: 48 m) | one bin (1-D) | the file's coordinate | 0 % | 909 / 909 | 3.26 / 3.25 | 575 (1394) / 572 (1404) | refused both | – |
| figure8_000552 | figure-8, both loops closed (DANDI 000552; no size stated) | 12.5 cm, measured | 6 nodes fitted; offset 1.00 cm | 1.0 % | 597 / 603 | 0.39 / 0.40 | 7.8 (56.8) / 8.8 (51.0) | 6, 14 / 0, 20 (21 trials) | 0.014 / 0.028 (workaround) |
| grosmark_000044_circular | 1 m diameter circle, 72-gon loop (Grosmark & Buzsaki 2016) | 33.9 cm, measured | robust circle centre (own radius 49.9 cm); offset 3.15 cm | 1.0 % | 1709 / 1726 | 0.66 / 0.83 | 6.5 (65.4) / 12.1 (64.9) | 67, 65 / 1, 85 (about 97 clockwise, G-037) | 0.055 / 0.095 (workaround) |
| hc18_train261 | closed circuit, 72-gon loop from the travel samples (hc-18: a model train; no shape stated) | 35.2 px, measured | sector medians; offset 2.86 px | 92.8 % (sleep-box sessions; 1 % of travel) | 1423 / 1441 | 0.18 / 0.19 | 275 (485) / 303 (513) px | 35, 31 / 1, 94 (about 107, G-037) | – |
| jadhav_000447 (novel W, runs 1 and 3) | W-track (Tang et al. 2023: ~80 x 80 cm, ~7 cm tracks) | 7 cm, stated | 6 nodes fitted; arms 85-101 cm; offset 2.12 cm, 64.1 % | 35.9 % | 441 / 912 | 0.55 / 0.58 | 10.0 (113) / 18.5 (95) | refused / 1 | – |
| jadhav_000447_familiar (run 2, new) | W-track (as above) | 7 cm, stated | 6 nodes fitted; arms 84-89 cm; offset 0.82 cm, 87.1 % | 12.9 % | 219 / 331 | 1.07 / 1.08 | 61.9 (133) / 40.3 (116) | refused / 0 | – |
| hc28_hpa | W-track (Jadhav et al. 2016: ~80 x 80 cm, 7 cm sections) | 7 cm, stated | 6 nodes fitted; arms 65-76 cm; offset 2.75 cm, 62.7 % | 37.3 % | 457 / 811 | 1.04 / 0.93 | 16.9 (56.2) / 16.0 (53.5) | refused both | – |
| agarwal_24757638 | 3-arm maze, parallel arms on a base (Figshare; Agarwal et al. 2026: "2.13m-long", span not stated) | 28.8 cm, measured | 6 nodes fitted; arms 180-185 cm; offset 1.03 cm | 1.0 % | 1025 / 1053 | 0.17 / 0.19 | 4.0 (215) / 5.7 (192) | refused both | – |
| shipley_30902972 | 8-arm radial maze, a star (Shipley et al. 2026: 80 x 10 cm arms, 24.7 cm platform) | 10 cm, stated | 9 nodes fitted; centre to arm end 86-99 cm (stated 92.35); offset 1.20 cm, 80.9 % | 8.8 % | 1067 / 1350 | 0.27 / 0.24 | 64.6 (99.7) / 51.9 (93.9) | refused both | – |
| olafsdottir_5566548 | Z-track (Olafsdottir et al. 2016: 10 cm runways, 190/220/190 cm) | 10 cm, stated | 4 nodes fitted; sections 165/204/159 cm; offset 1.56 cm, 92.4 % | 7.6 % | 1152 / 1266 | 0.83 / 0.84 | 66.8 (147) / 115 (142) | refused both | – |
| giocomo_001701 | X maze, two junctions and four arms (Aery Jones et al. 2026: "130-cm X-maze") | 5.68 cm, measured | 6 nodes fitted; arms 50 cm, centre 19 cm; offset 0.92 cm | 1.0 % | 956 / 967 | 0.12 / 0.13 | 2.5 (64.8) / 3.5 (62.9) | refused both | – |
| dyads_001845 | W-maze (Shukla et al. 2026: 80 x 80 cm, ~7 cm sections) | 7 cm, stated | 6 nodes fitted; arms 80-94 cm; offset 0.27 cm, 93.9 % | 6.1 % | 24 / 31 | – | – | refused both | – |

Every recording above is declared; none was refused and none stays
inferred. Two are declared with a layout no source sizes: hc-18's circuit
(the source says only that the rat rode a model train; the loop is the
samples' own and its verdict in the fidelity study is "not verified") and
001695's track (a linear track with wells at either end is stated; its
length is fitted).

### Choices that changed a recording

- **000447 split in two.** The novel and familiar W-tracks lie rotated 90
  degrees in one camera frame and their arms cross, so one environment with
  two components would put each maze's samples on the other's arms where
  they overlap. `jadhav_000447` is now the novel W-track (runs 1 and 3;
  positions outside them are NaN, its own trajectories, `analysis_epochs`
  its runs), and `jadhav_000447_familiar` is run 2, a new registry entry
  sharing the loader (`load_jadhav_000447.load_maze`). Tang, Shin & Jadhav
  2023, the paper DANDI lists as describing 000447, give one size for both
  mazes; the fitted mazes differ (novel arms 85-101 cm, familiar 84-89 cm,
  G-153).
- **The figure-8's positions are the camera (x, y).** A loop cannot be
  declared on the linearized line, where 0 and 222 cm are different points
  (G-021); the linearized coordinate moved to `params["linearized"]`, as on
  the other 2-D tracks, and the two range expectations now check it. The
  inferred run is therefore the camera grid (667 bins), not the old 1-D
  environment.
- **000165 is in cm.** Aery Jones et al. 2021 state "a 36 cm linear track";
  the tracked span is 35.4 cm along the axis, so the file's cm label is
  right and the loader no longer returns arbitrary units.
- **000044 linear: reward areas as end segments, and a 95th percentile
  width.** The stated 1.9 m maze is a 1.6 m stem with a 15 cm reward area
  continuing it at each end, placed along the line the file's linearized
  coordinate traces (0.963 cm of 2-D distance per linearized cm). The
  reward-end samples spread up to about 27 cm across the track's axis, and
  the first 110 s (3.5 % of samples) are an excursion 140 cm and more away;
  a 99th percentile width would make the track 3 m wide, so this loader
  measures the 95th (55 cm) and the excursion stays outside.
- **The labyrinth at one cell.** The task named one cell; the paper's
  corridors are 1-1/8 in on a 1.5 in grid, 0.75 of a cell, at which 18 % of
  nose samples would be off the track against 4.6 % at one cell.

## What changed between declared and inferred

- **Connectivity is the apparatus's.** Every declared track is one part
  (Takigawa two, as stated) with no link through a wall or a gap. In the
  fidelity study the labyrinth's path along the declared graph is 0.95 x
  the corridor distance where the samples' grid gives 0.66 (G-110),
  Takigawa's 0.97 where the grid gives 0.87, and the 000044 circle 0.96.
- **Decoding error falls** on most tracks (000044 linear 8.5 -> 2.5 cm,
  Takigawa 13.7 -> 7.8, Vollan 8.5 -> 5.0, the circle 12.1 -> 6.5, J16 5.7
  -> 2.0, 000447 novel 18.5 -> 10.0, the Z-track 115 -> 67, Hafting 157 ->
  106), because the decoder works on one coordinate along the track rather
  than a 2-D grid, and errors are along the track. Where many samples are
  off a stated width (J16, the W-tracks, hc-3) the declared error is over
  the samples on the track only. It rises on Shipley's maze (51.9 -> 64.6),
  000447's familiar maze (40.3 -> 61.9) and 000568 (32.1 -> 34.2); chance
  rises with it where the along-track distances are longer than the grid's
  (Shipley 93.9 -> 99.7, familiar 116 -> 133).
- **Laps exist on the loops.** `detect_laps`' default, which found 1 lap on
  the samples' grid of both 000044's circle and hc-18's circuit (G-037),
  finds 67 and 35 on the declared loops, and 6 on the figure-8 (0 on the
  grid). By start region the declared loops give 65, 31 and 14, the
  samples' grids 85, 94 and 20 (truth: about 97 clockwise laps, about 107,
  and 21 trials). The declared counts are low because 1 % of samples lie
  outside the measured width and each splits the recording (G-151): at a
  width with no sample outside, the figure-8 gives 20 and hc-18 92-97. So
  the declared loop fixes the default method's loop detection but not its
  undercount (G-037 stays open), and the region method is only as good as
  the share of samples on the track.
- **Replay runs on the track.** 000568 and 001695, which had no linearized
  coordinate and errored (G-038), now decode on the declared track (Grosmark
  rates 0 and 0.040); 000044 and the figure-8 no longer need the file's
  coordinate (status `ok` instead of `workaround`): 000044 linear 0.140 ->
  0.155, the circle 0.095 -> 0.055, the figure-8 0.028 -> 0.014. Davidson
  rates are 0 on all of them in both runs.
- **Spatial information is lower on most tracks** (000044 linear 0.81 ->
  0.38 bits/spike, Vollan 0.29 -> 0.15, Takigawa 2.22 -> 1.66, the circle
  0.83 -> 0.66), and the same or higher on the mazes with a stated width
  (hc-28 0.93 -> 1.04, Shipley 0.24 -> 0.27). A track bin pools positions
  across the track's width that the 2-D grid keeps apart, so the grid's
  extra information is partly lateral position and the head's swings at
  the reward ends; that is a hypothesis, not tested here.
- **Occupancy falls by the share outside.** J16 802 -> 424 s, 000447 novel
  912 -> 441 s, hc-28 811 -> 457 s, hc-3 535 -> 379 s, 000568 950 -> 819 s,
  Shipley 1350 -> 1067 s, with no warning from `occupancy` or
  `compute_spatial_rates` (G-155; G-152 for why).
- **Area-only scores are n/a on a track** (grid, border, egocentric
  boundary). On the samples' grids the border score refused on 19 of the 22
  (G-005) and the grid score had a finite value on 5.

## Gaps

- G-150: `from_graph`, `linear_track` and `maze` document nearest-track
  placement without a width; `bin_at` refuses it (docs).
- G-151: one sample outside a track's width splits `detect_laps`'
  recording; at 1-5 % outside most laps are lost (figure-8, hc-18).
- G-152: at a stated width, 5-64 % of head positions are off the track;
  `track_width` is both the apparatus's width and the placement tolerance.
- G-153: fitted mazes against their stated sizes (the Z-track 0.84-0.93 of
  the stated sections; W-tracks 0.8-1.25 of ~80 cm).
- G-154: 000044's circular coordinate leaves out the 30 cm reward area,
  which explains G-039's 72 % "on track" and its 134 cm extent.
- G-155: occupancy and rate maps drop positions outside a declared track
  without a warning; `detect_laps` and `decoding_error` do warn.

Existing gaps touched: G-110 and G-113 have a harness answer now (the
labyrinth's graph, 000447 split); G-021 and G-038 for the figure-8 and
000044 (declared loops, replay on the track); G-037 (defaults now find
laps on a declared loop, still too few); G-016 (J16's and Hafting's graphs
now have widths).

## What could not be verified or done

- Widths: no source read states the width of 000044's tracks, Vollan's
  linear track, 000165, 001695, Hafting's track, Takigawa's tracks, the
  figure-8, Agarwal's maze or the X maze; those widths are measured from the
  samples. hc-31's layout (Rich et al. 2014 not readable) and hc-18's
  circuit (Drieu et al. 2018's supplement not readable) are not stated.
- What Agarwal's "2.13m-long" and the X maze's "130-cm" span is not stated,
  so neither length is compared; 001695's size is in a supplementary table
  that could not be read.
- hc-3's length is not an independent check: its pixel scale comes from the
  same 250 cm.
- The W-track and Z-track fits are of a stated topology with free node
  positions; the stated sizes are compared, not imposed (G-153), since
  imposing them would move the track off the samples.
- Lap counts by region depend on the measured width (G-151); the truths
  quoted are the unwrapped angle (G-037) and the file's trial table.
- The 000044 recordings were streamed again for every run, so their 15
  minutes per run were not shortened.
