# Gap log

One entry per finding, numbered in order of discovery. `Kind` is one of
wrong-result, missing-feature, friction, error-message, docs, performance,
data-quirk, false-alarm, silent-hazard. Results referred to below are in
`results/` (`matrix.md`, `warnings.md`, `reference_checks.md`, `<key>.json`).

## G-001: Spike-drop warning fires when spikes fall outside tracking but also outside the requested epochs

- Datasets: peyrache_mouse32, figure8_000552
- Operation / step: hd.tuning (`compute_directional_rates(..., epochs=wake)`); rates.spatial, decode.position, replay.compare_methods
- Kind: false-alarm
- Evidence: head direction is tracked only during the wake epoch (8812–10771 s);
  the recording runs 0–22025 s with sleep before and after. Passing
  `epochs=wake` still warns "3995515/4398008 spike_times (91%) across all
  neurons fell outside the position time window [8812.42, 10771.3] … Check that
  spike_times and times share units". The spikes are sleep spikes, not a unit
  error. The count ignores `epochs`: `src/neurospatial/encoding/_spatial_counts.py:676-690`
  counts every spike outside `[times[0], times[-1]]`. `compute_directional_rates`
  has no `warn_on_drop=` (`compute_spatial_rates` has one), so the only way to
  silence it is to trim the spike trains by hand. The same warning is what
  catches spike times in ms (hazard probes in `warnings.md`), so the fix must
  keep that. On figure8_000552 (tracking only during the maze epoch,
  7891–9483 s, of an 18,557 s recording) every spike-based operation warns
  "5053296/5637087 spike_times (90%) … fell outside the position time window",
  and `decode.position` warns twice for the one fact (position window and
  decode window).
- Harness workaround: none (the warning is recorded, the op is `ok`)
- Proposed change: count as dropped only spikes inside the requested
  `epochs ∩ spike_window` that fall outside tracking; keep the clock-ratio hint
  for the units case; add `warn_on_drop=` to the directional entry points.
- Status: fixed (#265): the count is epochs-aware and warns once per call; no neurospatial warning fires on the clean recordings now

## G-002: `method="binned"` smoothed with a 5 cm bandwidth unless `bandwidth=0` was passed

- Datasets: sargolini_000582 (reference check 1)
- Operation / step: reference check, spatial information vs pynapple
- Kind: docs
- Evidence: with `method="binned"` and no `bandwidth`, neurospatial's spatial
  information is 0.248 bits/spike for unit 0 against pynapple's 1.389 (median
  relative difference over the six units 0.63); `analysis_settings` shows
  `bandwidth=5.0`. With `bandwidth=0.0` the two agree within 2.1 % (unit 0:
  1.3896 vs 1.3893), the rest explained by spike assignment (G-003). A user
  comparing "binned" information with published unsmoothed values gets numbers
  several times too small.
- Harness workaround: none now; `datasets/reference_checks.py:_binned_rates`
  passed `bandwidth=0.0` until the default changed and now uses the default.
- Proposed change: `method="binned"` defaults to `bandwidth=0` (no smoothing)
  everywhere binned is accepted (maintainer decision).
- Status: fixed (UX fixes phase 3b, branch `fix/ux-defaults-and-returns`):
  `method="binned"` is unsmoothed unless `bandwidth` is passed; reference
  check 1 now reports that the default and `bandwidth=0.0` agree exactly.

## G-003: Spikes are assigned to the tracking sample at or before them; pynapple uses the nearest sample

- Datasets: sargolini_000582 (reference check 1)
- Operation / step: reference check, spatial information vs pynapple
- Kind: docs
- Evidence: per-bin spike counts: neurospatial matches the at-or-before rule
  exactly (0 of 4013 spikes in a different bin) and pynapple the nearest-sample
  rule (1 of 4013); the two rules place 218 spikes in different bins at 50 Hz
  tracking, which moves spatial information by up to 2.1 %. Occupancy also
  differs slightly: neurospatial sums sample durations (599.96 s, the last
  sample has none), pynapple counts samples at the sampling rate.
- Harness workaround: none
- Proposed change: state the assignment rule (`time_allocation="start"`) where
  rate maps are documented; no code change.
- Status: open

## G-004: No public path to decode head direction

- Datasets: peyrache_mouse32 (reference check 3)
- Operation / step: reference check, head-direction decoding
- Kind: missing-feature
- Evidence: `neurospatial.decoding` decodes position on an `Environment`; no
  heading decoder is public and no recipe in `docs/user-guide/` decodes head
  direction. pynapple's `decode_bayes` on the 31 ADn units (61 bins, 100 ms,
  wake) gives a median absolute error of 8.9° against the tracked heading over
  the 18283 bins outside tracking gaps longer than 0.2 s (9.5° over all 19590
  bins, the tutorial's count).
- Harness workaround: none
- Proposed change: undecided — a circular (ring) environment the existing
  decoder accepts, or a documented recipe.
- Status: open

## G-005: Border score refuses an environment built from samples

- Datasets: sargolini_000582, giocomo_001701
- Operation / step: cells.border_score
- Kind: friction
- Evidence: `rates.border_scores()` raises `NotImplementedError: Physical
  boundaries are unknown: this environment's outline was inferred from
  positions …` and names the fixes (declare walls with `from_polygon`, an
  `arena=`, or `boundary_source='analysis'`). The message is clear; the cost is
  that the first environment most users build cannot be border-scored. The
  source says "1 x 1 meter enclosure" but not where the walls are in the
  tracking frame, so the harness does not declare them.
- Harness workaround: none (cell is `err`)
- Proposed change: undecided — probably none beyond surfacing the requirement
  in the border-score docs and the quickstart.
- Status: open
- More evidence (archive batch): the same refusal on hafting_2008_nird, takigawa_7114733, olafsdottir_5566548, shipley_30902972, agarwal_24757638 and j16_23020757; none of these sources gives wall positions in the tracking frame. `rates.egocentric_boundary` meets the same requirement and runs with `boundary_source="analysis"` (G-082).
- More evidence (NWB breadth batch): the same refusal on all eight new
  recordings with spikes and 2-D positions (000987, 001880, 001695, 001754,
  000568, 000165, 000003, Kanter): every environment there is built from
  samples.
- More evidence (faithful arenas batch): with the stated outline declared
  (`Environment.from_polygon`), `border_scores()` runs on 000582, Gardner,
  Vollan's open field, Kanter and hc-2 (cells `ok`); 000987, 001880,
  000003 and Hafting stay built from samples and still refuse. On the
  samples-built environments, `boundary_source="analysis"` would make the
  edges of unvisited holes walls: 515 of 574 Vollan units, 258 of 299 Kanter
  units and all 34 hc-2 units then score above 0.5. Declared walls bring
  their own failure where the head never reaches the wall bins (G-131).

## G-006: Grid score differs from opexebo on the same rate map because of the autocorrelogram

- Datasets: sargolini_000582 (reference check 2)
- Operation / step: reference check, grid score vs opexebo
- Kind: wrong-result (candidate; which convention is right is not settled)
- Evidence: on the same unsmoothed 40 × 40 maps, grid scores differ by up to
  0.381 (unit 2: neurospatial 0.179, opexebo 0.560), Pearson r 0.737. The
  scoring step agrees: neurospatial's `grid_score` on opexebo's autocorrelogram
  is within 0.012 of opexebo for every unit. The autocorrelograms differ:
  neurospatial leaves 2822 of the 5041 lags in opexebo's extent undefined (NaN
  where fewer than a quarter of the visited bins overlap), and at shared lags
  the values differ by up to 0.178. Two conventions account for it: opexebo's
  `autocorrelation` replaces NaN with 0 (unvisited bins as 0 Hz) — zero-filling
  the map on the neurospatial side moves unit 2 from 0.179 to 0.307 — and the
  undefined-lag rule accounts for the rest (0.307 vs 0.551 on opexebo's
  autocorrelogram). A threshold near 0.3–0.4 would classify unit 2 differently.
- More evidence (populations): on Gardner 2022 (483 module units) and
  Vollan 2025 rat 26018 (233 units in three modules), `reference_checks.md`
  sections 2b–2c score each unit under every convention. A NaN-aware Pearson
  at every lag with at least 20 overlapping bins (option a) is closest to
  opexebo on Gardner (r 0.966 vs 0.886 for today's rule) and closest to the
  authors' own module medians on Vollan: module 3, the largest spacing,
  scores 0.16 with today's quarter-overlap rule against the authors' 0.68
  (option a 0.77, opexebo 0.48). The quarter-overlap rule over the full lag
  range reproduces today's scores exactly, which checks the harness version.
- Harness workaround: none
- Proposed change: undecided — compare the overlap rule and NaN handling with
  BNT/opexebo's `normxcorr2_general`, then either match it or document why not.
- Status: open

## G-007: Position decoding with six MEC units is at chance; silent bins all decode to one corner

- Datasets: sargolini_000582
- Operation / step: decode.position
- Kind: docs
- Evidence: median error 66.0 cm with 250 ms bins in a 100 cm box, against a
  chance level of 55.3 cm (`chance_median_error` in the results: true position
  of each decoded bin vs that of a random other decoded bin, seeded). 40 % of bins have no spikes; every one of
  them decodes to (−47.5, 47.5), next to the bin with the lowest summed rate
  (−50, 50, summed rate 0) — the Poisson likelihood of silence, so the decoder
  is behaving as specified. Error falls to 47.1 cm at 2 s bins. Nothing in the
  result says it is uninformative.
- Harness workaround: none
- Proposed change: undecided — report a chance-level baseline with
  `error_against`, or point to `posterior_entropy` in the decoding docs.
- Status: open

## G-008: Mouse32 head direction has no stated reference frame and the file has no x/y position

- Datasets: peyrache_mouse32
- Operation / step: loader
- Kind: data-quirk
- Evidence: `acquisition/CompassDirection/ry` has `reference_frame=''`, unit
  "radian", values in [0, 2π). Where 0 points and which way angles increase is
  not stated, so the loader does not rotate it. The file has no `Position`
  container anywhere in the tree (walked with `h5py.visititems`).
- Harness workaround: `datasets/load_peyrache_mouse32.py` wraps to [−π, π] only
- Proposed change: none for neurospatial; `read_head_direction` could return
  the empty frame so callers can see it is unknown (it returns `frame=None`).
- Status: open

## G-009: 000582's tutorial session has one LED, so head direction cannot be computed

- Datasets: sargolini_000582
- Operation / step: loader
- Kind: data-quirk
- Evidence: only `processing/behavior/Position/SpatialSeriesLED1` exists in
  `sub-10884_ses-03080402` (full tree walked). `hd.tuning` is `n/a` for this
  dataset.
- Harness workaround: none
- Proposed change: none; a later batch can use a two-LED 000582 session.
- Status: open

## G-010: Spike times in ms pass unnoticed when the tracking covers only part of the recording

- Datasets: peyrache_mouse32 (hazard probe `spike_times_in_ms` / hd.tuning), figure8_000552 (`spike_times_in_ms` / rates.spatial)
- Operation / step: hd.tuning
- Kind: silent-hazard
- Evidence: with spike times multiplied by 1000, `compute_directional_rates`
  emits only the spike-drop warning the clean run already emits (91 % → 100 %
  of spikes outside the tracking window, same message), and no clock-ratio
  hint: the spikes span the whole 22025 s recording while tracking covers the
  1959 s wake epoch, so the ratio rule does not fire. The result changes
  quietly: median mean vector length 0.436 → 0.389, and 10 of 49 units lose a
  finite value. On 000582, where tracking covers the whole recording, the same
  hazard is caught with "divide them by 1000". On figure8_000552 the same
  hazard is silent too: median peak rate 4.67 → 0.00 Hz with only the
  clean-run warning.
- Harness workaround: none
- Proposed change: compare the spike span with the span of the requested
  epochs (or the whole tracked session) rather than with `times` alone; fixing
  G-001 removes the clean-run warning that hides this case.
- Status: fixed (#265): with the clean-run warning gone, the spike-drop warning flags the ms case on Mouse32 and the figure-8
- More evidence (NWB breadth batch, after #265): spike times x 1000 are
  silent on 2 of 9 new recordings. On 001695 (no `epochs`; sleep spikes
  outside the maze tracking) the clean run already emits the spike-drop
  warning for 75 % of spikes, and the hazard only raises it to 100 % with the
  same message and no span ratio: median peak 0.73 -> 0.0 Hz. On 000165
  (`epochs` 0-269.6 s) nothing warns: 883 of the 557,063 ms-scaled spikes
  still fall inside 0-269.6 s, and the rest lie outside the epochs, which
  the epochs-aware drop count does not count: median peak 4.58 -> 0.03 Hz.
  The other seven are caught: on 001754 and 000003 (`epochs` passed, but the
  tracking spans the whole recording) the drop count against the tracking
  window gives "about 1000 times ... divide them by 1000"; on 000568 and
  Kanter the only new message is "All N units are silent ... pass
  spike_window", whose advice is wrong for a unit error.
- Proposed change (addition): apply the span-ratio test (spike span about
  1000 times the analysed span) to the analysed span, the requested `epochs`
  or else the tracking, not only to `times`. A rule "warn when no spike falls
  inside the epochs" would miss 000165, where 883 ms-scaled spikes still do.

## G-011: `read_intervals` does not search processing modules

- Datasets: figure8_000552
- Operation / step: loader (`read_intervals`)
- Kind: missing-feature
- Evidence: the file keeps its ripples (`processing/ecephys/Ripples`, 4297
  rows) and sleep states (`processing/behavior/SleepStates`, 27 rows) as
  `TimeIntervals` in processing modules. `read_intervals(nwbfile,
  interval_name="Ripples")` raises "TimeIntervals 'Ripples' not found …
  Available intervals: ['trials', 'epochs']". `read_position` searches
  `processing/*` and `acquisition`; `read_intervals` only `intervals/`.
- Harness workaround: `datasets/load_figure8_000552.py` reads both tables with pynwb
- Proposed change: search processing modules (and list what was found there)
  as `read_position` does.
- Status: open

## G-012: `read_position(position_name=)` rejects the `Position` container name

- Datasets: figure8_000552, grosmark_000044_circular, grosmark_000044_linear
- Operation / step: loader (`read_position`), io.nwb_readers
- Kind: error-message
- Evidence: with two position series, `read_position(nwbfile)` refuses to
  choose (good). `position_name="LinearizedPosition"` (the container's name)
  then raises "Position series 'LinearizedPosition' not found … pass
  position_name= with a name or path from this list", and the list holds only
  full series paths. The container name is what a user sees first in the file.
  With default arguments (`io.nwb_readers`) the refusal is the right
  behavior; the cell is an error because a reader raised, as the operation
  defines it. 000044 repeats it: `position_name="CircularMazePosition"` is
  refused, and the default call lists the linearized series as a second
  Position candidate.
- Harness workaround: the loader passes the full series path
- Proposed change: accept a `Position` container name when it holds one series.
- Status: open
- More evidence (NWB breadth batch): 001880 lists `SubjectPosition/PositionSeries`
  and `SubjectPosition/VelocitySeries` (vx, vy, speed in cm/s) as two
  Position candidates; 001845 lists six processed keypoints. Both refusals
  are right; the loaders pass the full series path.

## G-013: `env.occupancy(min_speed=)` needs `speed=`; the rate and decoder entry points compute it

- Datasets: figure8_000552, rosenberg_labyrinth, yartsev_bat_demo
- Operation / step: env.occupancy
- Kind: friction
- Evidence: `env.occupancy(times, positions, min_speed=4.0)` raises
  "min_speed=4.0 was given without speed … pass speed= (one value per sample,
  for example compute_speed(times, positions)), or min_speed=None", while
  `compute_spatial_rates(..., min_speed=4.0)` and `BayesianDecoder.fit(...,
  min_speed=4.0)` on the same arrays compute speed themselves and run. The
  same argument means different things across siblings.
- Harness workaround: none (cell is `err`)
- Proposed change: compute speed in `occupancy` as the rate functions do (or
  require `speed=` everywhere).
- Status: fixed (UX fixes phase 3b, #262): `env.occupancy(min_speed=)` now derives speed like the rate functions; the occupancy cells are `ok` on all three datasets

## G-014: A length unit off by 100 on a 1-D track builds 3 bins without a warning

- Datasets: figure8_000552, yartsev_bat_demo (hazard probe `length_unit_off_by_100`)
- Operation / step: env.from_samples, rates.spatial, behavior.segment_by_velocity
- Kind: silent-hazard
- Evidence: positions ×0.01 (metres labelled cm) on the 222 cm track give a
  2.2 "cm" track: `Environment.from_samples(bin_size=2)` builds 3 bins with no
  warning, rate maps flatten (median peak 4.67 → 1.36 Hz) and velocity
  segmentation finds 0 running segments instead of 395, all silently. On
  000582 the same hazard warns because the bin is at least the whole extent
  in every dimension; here the extent (2.2) is just above the bin (2.0). On
  the bat (cm labelled m, positions ×100) the environment refuses a 9e9-bin
  grid, but `segment_by_velocity` with the file's 0.5 m/s threshold silently
  finds 668 movement segments (882 s) instead of 69 (146 s): speeds 100 times
  too high pass any threshold. With the speed gate in `env.occupancy` (G-013
  fixed), the figure-8's occupancy above 4 cm/s silently falls from 521.8 s
  to 14.0 s: speeds 100 times too low fail it.
- Harness workaround: none
- Proposed change: warn when the environment has fewer than a handful of bins
  along a dimension (the existing check only fires at one bin or fewer).
- Status: open
- More evidence (CRCNS batch): on hc31_anm190422's 48 m track (1-D), ÷100
  gives a 48 "cm" track of 21 bins, above check 2's threshold of 5; nothing
  in the environment can tell, and `env.occupancy` and `rates.spatial` refuse
  it only through `min_speed` (every interval excluded, with the units hint).
  Check 2 does not cover a long track; the speed gate does.
- More evidence (archive batch): `length_unit_off_by_100` is caught by `Environment.from_samples` on all seven new cm recordings (bin at least the extent), but `segment_by_velocity` finds 0 running segments silently on each (281, 604, 491, 715, 516 and 290 -> 0; 668 -> 1 on the Z-track), and on hafting_2008_nird and olafsdottir_5566548 occupancy and rate maps change without a warning of their own (occupancy 655.6 -> 0.24 s and 1266 -> 0.84 s).
- More evidence (NWB breadth batch): `segment_by_velocity` with positions
  /100 finds 0 running segments, silently, on 001880 (1,030 clean), 001695
  (315), 000568 (458) and 001845 (10); the environment and rate steps on the
  same recordings warn or raise.

## G-015: `detect_laps` finds laps on a maze with no loop

- Datasets: rosenberg_labyrinth, sargolini_000582, miniscope_002030, jadhav_000447, vollan_of
- Operation / step: behavior.laps
- Kind: wrong-result
- Evidence: the labyrinth is a binary tree (`nx.is_tree` on the track graph is
  True). Its `Environment.from_graph` bin graph has 412 edges for 350 bins and
  63 three-bin cycles, one triangle at each junction. `detect_laps(env, times,
  positions)` (method `auto`) accepts it as a loop environment and returns 3
  "laps" of 0.9–1.3 s covering 6–7 bins each (one clockwise, one
  counter-clockwise, one unknown). Its docs say `auto` raises when "its
  connectivity graph has no cycle"; the cycles here are the junction cliques,
  not loops of the maze. On sargolini_000582, an open 1 m box with no track,
  `detect_laps` likewise accepts the grid's bin cycles as a loop and reports
  one clockwise "lap"; on miniscope_002030 (a 96 cm circular open field) it
  reports one counter-clockwise lap.
- Harness workaround: none
- Proposed change: test for a loop on the track graph (or ignore cycles that
  stay within one junction's bins) before counting laps.
- Status: partly fixed (#265): mazes and 3-D spaces are refused; open arenas
  (000582 box, 002030 arena) still give one lap each, apparently because
  unvisited patches inside an arena built from samples count as holes
- More evidence (second batch): on the open arenas, Gardner's 1.5 m box is
  refused ("no loop"), Vollan rat 26018's box gives 0 laps (1 once the
  speed-filter gaps are closed, hazard `times_rebuilt_from_dt`), and the
  000447 W-track, a track with no loop built from samples, gives 1 clockwise
  lap. Straight tracks (000044 linear, Vollan linear) are refused, as they
  should be. On a real loop the opposite fails: G-037.
- More evidence (CRCNS batch): hc-2's 180 cm open field gives 1 clockwise
  "lap" (hazards that move untracked frames to a corner flip it to
  counter-clockwise); the hc-28 W-track is refused ("no loop"), while the
  000447 W-track gave 1 lap; the hc-3 250 cm linear track and the hc-31 48 m
  track are refused, as they should be.
- More evidence (archive batch): refused ("no loop") on the water maze, the two Takigawa tracks, the Z-track, the radial maze and the 3-arm maze, and on the 3-D lattice ("cannot tell whether this 3-D environment has a loop"); none of these has a loop, so the refusals are right. The water-maze pool is an open arena that is refused, not given a spurious lap.
- More evidence (NWB breadth batch): refused ("no loop") on the 000987 and
  001880 arenas, the 001695, 000568 and 000165 linear tracks, the 001754
  staircase session (all five sessions' tracks in one image), the 001845
  W-maze and the 000003 arena, as it should be. Kanter's 1 m open field
  gives 1 clockwise "lap" (cell `ok`): the open-arena case that #265 left.
- More evidence (environment fidelity, `studies/environment_fidelity.py`):
  the unvisited patches are counted. Holes (unvisited cells that visited
  bins enclose) in the driver's environments: 000582 72 holes (136 bins, a
  10 min session), 002030 19 (722 bins: the mouse ran along the wall and
  crossed the 96 cm arena on a few paths, so bins cover 37 % of the stated
  circle), Kanter 49 (460 bins, 65 % of the 1 m square covered), 000405 105
  (446), Vollan's open field 131 (297), the water maze 8 (9); none in
  Gardner's (143 min), 001880's (287 min) or 000987's (20 min).
  `Environment.open_field` (the same with `fill_holes=True`) removes them:
  000582 keeps 2 holes and its bins then cover 94 % of the 1 m square,
  002030 none (99 % of the circle), Kanter none (94 %), 000405 none. Around
  the holes the outlines match the stated squares: the outermost bins come
  within a median 1.6 cm (0.6 bin) of the stated wall in 000582, Kanter and
  Gardner, so border and boundary measures see the wall where it is.
- More evidence (faithful arenas batch): with the stated outline declared,
  `detect_laps` refuses every open arena ("no loop"): 000582, 002030,
  000405, Vollan's open field, Kanter and hc-2, which gave 0-1 spurious laps
  on the samples-built environment (`results/inferred_env/`). The refusal is
  right; the holes of a samples-built arena were the loops. On the 3-D
  lattice declared as its bars it finds a lap instead (G-134).

## G-016: A track graph without `track_width` cannot place positions

- Datasets: rosenberg_labyrinth
- Operation / step: behavior.laps, behavior.turn_sequence
- Kind: friction
- Evidence: `Environment.from_graph(graph, edge_order, bin_size=0.5)` builds,
  but `detect_laps` and `graph_turn_sequence` then raise "Positions cannot be
  assigned to the bins of a track built without track_width", with a clear fix
  (`env.with_track_width(width)`). The source states the corridor width (one
  maze cell), and with `track_width=1.0` both run.
- Harness workaround: none (the loader states `geometry["track_width"]` from the source)
- Proposed change: none beyond showing `track_width=` in the `from_graph`
  example; the refusal is the right behavior.
- Status: open
- More evidence (archive batch): hafting_2008_nird (a one-edge 320 cm track) and j16_23020757 (the authors' 10-node bandit-maze graph) build with `from_graph` (80 and 248 bins) but `detect_laps` and `graph_turn_sequence` refuse both, because neither source states the track width.

## G-017: Missing positions stored as 0 silently change movement segments and turns

- Datasets: rosenberg_labyrinth, yartsev_bat_demo, giocomo_001701 (hazard probe `missing_positions_as_zero`)
- Operation / step: behavior.segment_by_velocity, behavior.turn_sequence
- Kind: silent-hazard
- Evidence: 4.9 % of the labyrinth's frames have no nose keypoint (NaN). With
  them set to (0, 0), a corner of the maze, velocity segmentation finds 2936
  movement segments instead of 2637 and the turn sequence 4604 turns instead
  of 4415, with no warning. On 000582 the same hazard touches one sample and
  is undetectable; here the zeros are 12,400 samples at exactly (0, 0). On
  the bat (0.24 % missing), the zeros land at (0, 0, 0), below the lowest
  tracked height (0.28 m): the environment loses 3 bins and the median mean
  rate in flight drops 4 % (0.945 → 0.909 Hz), with no warning. With the
  speed gate in `env.occupancy`, the labyrinth's occupancy above 1 cell/s
  rises from 4,808 s to 5,115 s, the jumps to (0, 0) counting as movement.
- Harness workaround: none
- Proposed change: warn when a sizeable share of samples sit at exactly the
  same coordinate (or at exactly 0) between moving samples.
- Status: open
- More evidence (CRCNS batch): hc-28's `rawpos` stores 0 for every
  coordinate of an untracked frame (the documentation says so): 15.5 % of HPa
  day 1 epoch 4. The loader makes them NaN; put back (the generic hazard, which
  here restores exactly the file's zeros) they sit at the pixel origin outside
  the W-track: extent 92.5 → 145 cm, total occupancy up 158 s, of which the
  origin bin holds 175.3 s (the most of any bin),
  running segments 487 → 455, all silent. Check 1 would fire (15.5 % at the
  origin against 0 % on every clean recording, now 19 of 19 with positions).
- More evidence (archive batch): NaN positions as (0, 0) are silent on the water maze (0.29 % of samples), Takigawa (85 %: everything off the tracks), the Z-track (29 %: largest occupancy bin 43.9 -> 290.6 s) and the 3-arm maze, both as NaN -> 0 and as the file's own `xMM = yMM = 0` sentinel (0.20 %). Check 1 (origin share >= 0.1 %) would fire on all four.
- More evidence (NWB breadth batch): silent on 001880 (5.3 % NaN, mostly
  off the maze: 1,080 -> 1,085 bins, 1,030 -> 1,035 movement segments),
  001845 (38 % NaN: 245 -> 248 bins, 10 -> 14 segments), 001754 (8.8 % NaN:
  most-occupied bin 34.6 -> 115.6 s) and 000003 (0.04 % NaN: environment
  extent 168 -> 500 px). Check 1 (origin share) would catch each. 001754 also
  has its own lost-tracking code at (0, 255) (G-053), which an origin rule
  does not see.

## G-018: Replay methods run on one-time-bin events without a warning

- Datasets: figure8_000552
- Operation / step: replay.compare_methods
- Kind: silent-hazard
- Evidence: the file's ripple windows inside the maze epoch (71) have a median
  duration of 29 ms (10th–90th percentile 22–65 ms), so most are one 20 ms
  time bin. `DAVIDSON_2009` calls 0/71 replay (95 % CI 0–0.051) and
  `GROSMARK_2016` 2/71. Both recipes' notes say candidate events last at
  least 100 ms and that event selection is not applied; nothing warns that a
  line fit or weighted correlation is being run on a single time bin.
- Harness workaround: none
- Proposed change: warn (or refuse) when an event has fewer time bins than a
  recipe's score needs, and name the recipe's event-duration rule.
- Status: open

## G-019: The pre-registered node count left out the maze entrance

- Datasets: rosenberg_labyrinth
- Operation / step: expectations
- Kind: data-quirk
- Evidence: the expectation "127 nodes" (the article's count) fails: the track
  graph has 128, the article's 127 run ends plus the entrance where the mouse
  comes in from the home cage, which the corridor geometry needs. "63
  junctions" passes.
- Harness workaround: none (the expectation stays as registered)
- Proposed change: none
- Status: won't fix (the expectation, not the loader, was incomplete)

## G-020: A mirrored frame cannot be detected, and turn directions depend on it

- Datasets: rosenberg_labyrinth (hazard probe `image_y_down`)
- Operation / step: behavior.turn_sequence
- Kind: docs
- Evidence: with y left pointing down (image rows), every left turn becomes a
  right turn; the share of left turns barely moves (0.5001 → 0.4999) because
  the maze is symmetric, so no value check can catch it.
- Harness workaround: the loader flips y to point up
- Proposed change: say in `graph_turn_sequence` that left and right assume a
  y-up (right-handed) frame and that image coordinates must be flipped first.
- Status: open
- More evidence (archive batch): on the Z-track the y flip undone (hazard `image_y_down`) leaves every summary identical; the frame cannot be checked without a heading (one LED) or a stated layout.

## G-021: No periodic 1-D track: a linearized loop is treated as a line

- Datasets: figure8_000552, grosmark_000044_circular
- Operation / step: behavior.laps, behavior.segment_by_velocity, env.from_samples
- Kind: missing-feature
- Evidence: the file's linearized position runs around the figure-8 as a loop
  (G-026): 0 and 222 cm are the same place. A 1-D environment built from it is
  a line, so `detect_laps` raises "this environment has no loop … use
  detect_trials", the two ends are 222 cm apart for decoding and replay, and
  57 of the 62 wraps fall inside `segment_by_velocity` running segments as
  speed spikes of about 6,500 cm/s. `Environment.from_graph` cannot express
  the loop either: its node positions must be in the positions' coordinates,
  and on a 1-D axis 0 and 222 are different points.
- More evidence: the 1 m circular track of 000044 (Achilles 11012013)
  stores a linearized coordinate from 0 to 290 cm that wraps 123 times; its
  replay cell decodes on it as a line (G-038).
- Harness workaround: none (laps cell is `err`)
- Proposed change: undecided — a periodic 1-D track (a length and a flag that
  the ends meet), or a documented recipe that embeds a linearized loop as a
  circle for `from_graph`.
- Status: open

## G-022: In sparse 3-D occupancy, spatial information is at the shuffle level while its bias estimate is small

- Datasets: yartsev_bat_demo
- Operation / step: rates.spatial_information, rates.information_vs_shifts, cells.place_screen
- Kind: wrong-result (candidate; needs a full shuffle test)
- Evidence: 3-D rate maps of 289 units over 1352 bins of 0.2 m, restricted to
  37 flights during the neural recording (121 s of occupancy; median 0.067 s
  per visited bin, 70 % of the 1201 visited bins under 0.1 s). Median spatial
  information is 3.47 bits/spike; with each unit's spikes circularly shifted
  within the counted flight time (4 seeded shifts,
  `rates.information_vs_shifts`) it is 3.18, and 57 % of units beat every
  shift (all 6 on 000582, 96 % on the figure-8). Before the shift was
  confined to counted time (G-043) the shifts gave 3.55 and 34 %.
  `spatial_information_bias`, the "expected untuned information" that
  `classify()` compares with, has a median of 0.37 bits/spike. The 101 units
  firing above 2 Hz during flight (which the record's script sets aside as
  putative interneurons) have a median of 1.36 bits/spike, the rest 4.62, so
  the high values come from sparsely firing units on sparse occupancy.
  `min_occupancy` defaults to 0 s.
- Harness workaround: none
- Proposed change: undecided — test `spatial_information_bias` against a
  shuffle null on sparse occupancy; warn when most visited bins hold very
  little time, and point to `min_occupancy` or larger bins.
- Status: open
- More evidence (CRCNS batch): hc18_train261, positions in pixels while a
  model train carries the rat at about 4.2 px per 25.6 ms frame (median),
  close to the 5 px bin: 341 of 2487 visited bins hold a single frame, and
  the median unit's peak rate is exactly 1 spike / 1 frame = 39.06 Hz.
  `min_occupancy` defaults to 0 s.
- More evidence (archive batch, rerun with shifts within counted time, G-043): the Grieves lattice (its declared bar graph, 4 cm bins, 2506 visited bins, 13 % of them under 0.1 s, median 0.62 s) gives the information-bias study's second sparse 3-D case. Untuned Poisson cells: mean information / mean bias 1.00, 1.01, 1.00 at 0.5, 2, 8 Hz (200 cells each); bursty cells 3.3, 4.3, 4.7. Real units (16 with a defined bias, 100 shifts each): median information 0.602, bias 0.182, shift-null mean 0.438, null / bias 2.56; 6 % pass the bias rule but not the shifts' 95th percentile. Bat: 0.92 (Poisson) and 2.22 (bursty) at 8 Hz; real units' null / bias 1.23 and 15 % pass the bias rule but not the shifts (2.19 and 44 % with the old whole-span shifts). The estimate holds for Poisson cells on sparse 3-D occupancy; the gap to the shift null is temporal structure the shifts keep (bursts), not firing locked to flights, which shifts within counted time do not keep.

## G-023: 3-D position decoding is near chance on the bat session

- Datasets: yartsev_bat_demo
- Operation / step: decode.position
- Kind: docs
- Evidence: training on even flights and testing on odd ones during the
  neural recording (250 ms bins, 0.2 m bins), the median error is 2.18 m
  against a chance level of 2.34 m.
  Likely related to G-022 (rate maps from sparse flight occupancy); not
  investigated further.
- Harness workaround: none
- Proposed change: undecided (revisit after G-022)
- Status: open
- More evidence (archive batch): on the Grieves lattice (24 units, 4 cm bins in 3-D) the median decoding error is 82.7 cm against a geodesic chance of 76.8 cm.

## G-024: Grid and border scores refuse 3-D environments

- Datasets: yartsev_bat_demo
- Operation / step: cells.grid_score, cells.border_score
- Kind: friction
- Evidence: `grid_scores()` raises "needs a regular 2D grid of rectangular
  cells … Fix: use spatial_autocorrelation_radial(env, firing_rate) … a radial
  profile does not test for sixfold symmetry"; `border_scores()` raises "Wall
  geometry needs a 2-D environment; this one is 3-D". Both messages say what
  to do; 3-D grid and border measures are a research question, not a missing
  convenience.
- Harness workaround: none (cells are `err`)
- Proposed change: none
- Status: won't fix (behaves as documented)
- More evidence (archive batch): grid and border scores refuse the Grieves lattice (3-D) the same way.

## G-025: 3.6 % of the bat's positions lie outside the room bounds the file states

- Datasets: yartsev_bat_demo
- Operation / step: loader
- Kind: data-quirk
- Evidence: 23,454 of 642,724 samples have y outside `r_lim` [−2.6, 2.6] m
  (down to −4.23 m); x and z are inside. Only 17 of them fall in flights,
  so they look like resting positions beyond the flight volume rather than
  tracking errors.
- Harness workaround: none (positions kept as recorded; rate maps use flights)
- Proposed change: none
- Status: open
- More evidence (environment fidelity): Forli et al. 2025 (Methods) give
  the indoor flight room as 5.6 x 5.2 x 2.5 m with two feeders on the wall
  at one end; `r_lim` spans 5.8 x 5.2 x 2.3 m, so its y range is the room's
  width, and the samples down to y = -4.2 m lie outside the room the paper
  describes as well. The 0.2 m voxel environment from
  all samples has 4.4 % of its 1,352 voxels outside `r_lim`: a corridor of
  voxels runs out of the room to y = -4.2 m (1.77 m past the wall) at
  x = -2.6 m, z = 0.9 m, where 2.4 % of the resting samples sit. 97.6 % of
  the samples are rest, mostly at one perch on the -x wall
  (-2.7 to -2.4, 1.5, 1.8 m). Of the two feeders `NP_Bat_Demo.m` places
  (`r_fd`, both at x = 2.8 m, the +x wall: the paper's "wall at one end"),
  one lies in a voxel and the other 0.17 m from the nearest voxel centre,
  outside the environment (the script itself corrects `r_fd` to the nearest
  rest cluster within 0.2 m). The voxels reach the x walls and the
  ceiling (within 0.11 m) but stop 0.12 m above the floor. The room's walls
  are not part of the environment, only the visited voxels.
- More evidence (faithful arenas batch): the loader now declares the room,
  every voxel of a grid tiling `r_lim` exactly (8,294 voxels, 29 x 26 x 11
  cells of 0.2 x 0.2 x 0.209 m; `from_samples` with `dimension_ranges=r_lim`
  and `infer_active_bins=False`), so the corridor of voxels to y = -4.2 m is
  gone (0 % of voxels outside the room, against 4.4 %) and both feeders lie
  in the environment. The 3.66 % of samples outside `r_lim` are not placed
  (`env.declared` `frac_samples_outside`). The fidelity study first judged
  that a mismatch (limit 2 %); with one verdict for every declared
  environment (`_geometry.judge_declared`) it is a tracking and room
  measure, reported and not judged: 0.11 % of the flight samples the
  analyses use lie outside `r_lim`, split by speed 0.0 % below and 0.1 %
  above the loader's 0.5 m/s `v_th`. The room is judged on stated facts
  only, `r_lim`'s sides against the paper's room (x +3.6 %, y +0.0 %,
  z -8.0 %, within 15 %): matches. Forli et al. 2025 state the indoor room as 5.6 x
  5.2 x 2.5 m and an outdoor enclosure 10.1 m long; `r_lim` and the
  samples' 5.7 m span place this session indoors. The paper's size centred
  on `r_lim` would leave 3.77 % outside, cutting samples at the x walls
  (they reach x = +-2.85 m), so `r_lim` is used. Decoding is worse on the
  room (median 2.36 m against 2.18 m on the visited voxels; chance 2.56 and
  2.64 m): 14 % of the room's voxels are visited in flight, and the uniform
  prior covers the rest.
- More evidence (review round): a room is now judged as an outline is, per
  face (`_geometry.judge_declared`: each judged sample assigned to the face
  it lies farthest past, inside the nearest; the 95th percentile of a
  face's samples past it more than one bin is a mismatch). On the flight
  samples, 57 are nearest the y-low face and 17 of them lie 1.4 m past it,
  at the perch beyond `r_lim`, so that face's 95th percentile is 1.41 m
  against a 0.2 m limit: the room is a mismatch. The sizes still match
  (x +3.6 %, y +0.0 %, z -8.0 %).

## G-026: The figure-8's linearized position wraps around

- Datasets: figure8_000552
- Operation / step: loader
- Kind: data-quirk
- Evidence: the linear coordinate jumps by a median of 215.6 cm 62 times while
  the camera position moves a median of 0.69 cm (at most 1.99 cm) across the
  same samples; samples below 3 cm and above 219 cm sit at the same camera
  point, (85.3, 43.3) ± (2.3, 1.7) cm. The file's description says only
  "Linearization of the (x,y) coordinates".
- Harness workaround: none (kept as stored; consequences in G-021)
- Proposed change: none for neurospatial beyond G-021
- Status: open

## G-027: `DAVIDSON_2009` verdicts depend on which other events are tested

- Datasets: simulated linear track (offline test construction)
- Operation / step: replay.compare_methods
- Kind: docs
- Evidence: with 100 shuffles per null, the same simulated stationary events
  are called replay 7 of 8 times when tested together with 8 sweeps, and 0 of
  16 times when tested with stationary events only. Stationary events can be
  replay, and the recipe's line search includes zero-speed lines on purpose,
  so accepting them is not the issue; the issue is that an event's verdict
  changes with the other events in the call. The pseudo-event null draws from
  the whole set of candidate events (the recipe's note), so the null for an
  event depends on the batch. The replay guide documents that `DAVIDSON_2009`
  accepts stationary events, not that its verdicts depend on the batch.
- Harness workaround: none
- Proposed change: say in the recipe's notes and the replay guide that
  pseudo-event verdicts depend on the set of events passed together.
- Status: open

## G-028: The bat's first flight precedes the neural recording

- Datasets: yartsev_bat_demo
- Operation / step: loader
- Kind: data-quirk
- Evidence: flight 0 runs from −15.40 to −14.05 s; the earliest spike of any
  of the 289 units is at −12.68 s, with a population rate of about 800
  spikes/s afterwards. Left in, it adds occupancy with no spikes: 13 bins
  visited only then get rate 0 instead of NaN, and nothing warns (a silent
  1.35 s is below what neurospatial's silent-unit check flags).
  `NP_Bat_Demo.m` restricts the session to where behavior and ephys overlap.
- Harness workaround: `datasets/load_yartsev_bat_demo.py` clips the analysis flights to the first–last spike span
- Proposed change: none for neurospatial (`spike_window=` handles it when the recording span is known)
- Status: open

## G-029: No spatial map from a continuous signal (calcium imaging, binned rates)

- Datasets: miniscope_002030, giocomo_000405
- Operation / step: imaging.rates
- Kind: missing-feature
- Evidence: every public rate function (`compute_spatial_rates`,
  `compute_tuning_curve(s)`, the decoder) takes spike times; no entry point
  accepts ΔF/F, deconvolved activity or binned rates sampled on frames, and
  `docs/user-guide/` never mentions calcium imaging. The harness workaround
  (`imaging.rates_workaround`) builds maps two ways: for 000405's binned rates,
  an occupancy-weighted mean per spatial bin (46 cells, median peak 63 Hz);
  for 002030's ΔF/F, one event per transient (onset above median + 3 robust
  SD, end below median + 1 SD), passed to `compute_spatial_rates` as spike
  times (26 cells, median peak 0.58 events/s; counting every re-crossing of
  the onset level instead gave 0.97). Both throw away amplitude or rely on
  threshold choices the library cannot see, and the event rule has no
  published source. On simulated transients it recovers a field (map vs true
  field r ≈ 0.6–0.8, ≈ 0 for a time-shifted trace) only at modest event rates:
  when transients overlap (≫ 2 events/s with a 0.5 s decay) the trace never
  falls back and a whole field visit counts once.
- Harness workaround: `datasets/operations.py:imaging_rates_workaround`
- Proposed change: undecided — a value-weighted rate map (occupancy-normalized
  mean of a signal per bin) sharing the rate maps' bins, NaN rules and
  smoothing; and a documented recipe for ΔF/F.
- Status: open

## G-030: `read_head_direction` trusts a "degrees" label on values that are radians

- Datasets: giocomo_001701
- Operation / step: io.nwb_readers (`read_head_direction`)
- Kind: wrong-result
- Evidence: `CompassDirection/head direction` is labelled `unit="degrees"`
  with values in −3.140 to 3.142. `read_head_direction(nwbfile)` converts
  them as degrees and returns −0.055 to 0.055 rad, with no warning, so a
  user's head direction collapses to a 6° arc. Fed to
  `compute_directional_rates`, it makes every unit look like a perfect
  head-direction cell (G-031).
- Harness workaround: `datasets/load_giocomo_001701.py` reads the values with pynwb and keeps them as radians
- Proposed change: refuse (or warn on) a degrees-labelled series whose finite
  values all lie within ±π and span most of it, naming both readings.
- Status: open (UX-fixes declined it for its plan; follow-up PR after that plan merges)

## G-031: Head directions squeezed into a small arc pass unnoticed

- Datasets: giocomo_001701 (hazard probe `headings_read_as_degrees`)
- Operation / step: hd.tuning
- Kind: silent-hazard
- Evidence: with the headings `read_head_direction` returns (radians read as
  degrees, all within ±0.055 rad), `compute_directional_rates` reports a
  median mean vector length of 0.999 over the 320 units instead of 0.12, with
  no warning: every unit looks like a sharp head-direction cell. Its unit
  check catches the opposite error (degrees read as radians, the
  `headings_in_degrees` probes), not headings confined to a few degrees.
- Harness workaround: none
- Proposed change: warn when the tracked headings cover only a small part of
  the circle, naming a radians/degrees mix-up as a likely cause. The rule
  needs calibrating first: on a linear track headings cluster in two
  opposite directions, so a naive arc-coverage measure would misfire; a
  circular spread, or the share of direction bins with occupancy, measured on
  the survey's real datasets would make the threshold defensible.
- Status: open (UX-fixes declined it for its plan; follow-up PR after that plan merges)

## G-032: The NWB readers miss data stored outside the standard containers

- Datasets: miniscope_002030, giocomo_000405
- Operation / step: io.nwb_readers
- Kind: missing-feature
- Evidence: 002030 stores head direction as a bare `SpatialSeries`
  (`processing/behavior/headdirection`, radians, frame stated) next to its
  `Position`; `read_head_direction` searches only `CompassDirection`
  containers and raises "No CompassDirection data found". 000405 keeps
  position and head direction in one generic `TimeSeries`
  (`acquisition/track_data`, columns named only in the description) and its
  activity as binned rates; `read_position`, `read_head_direction` and
  `read_units` all find nothing.
- Harness workaround: both loaders read the series with pynwb
- Proposed change: accept an explicit path to any `SpatialSeries` or
  `TimeSeries` (with a column choice) in the readers; consider finding a
  `SpatialSeries` whose name says head direction.
- Status: open
- More evidence (NWB breadth batch): 000003 stores its two position sensors
  as bare `SpatialSeries` in `acquisition` (not in a `Position`
  container); `read_position` raises "No Position data found" (G-062).

## G-033: 000405's labels disagree with its values

- Datasets: giocomo_000405
- Operation / step: loader
- Kind: data-quirk
- Evidence: `track_data` is labelled `unit="mm"` but spans −62 to 60 by −5
  to 134 (a rat open field in cm, not 12 cm); its `rate` attribute is 0.02,
  the sampling period (`firing_rates`, the same length, has `rate=50`);
  `firing_rates` is labelled `unit="uv"`, and its values reach 3,079 Hz
  (median 0, 99th percentile 47.5) with almost no values on the 50 Hz grid a
  count per 20 ms bin would give, so they look smoothed; the session start is
  in the year 18. The column `hd` is not head direction: it equals the
  direction of movement computed from x, y (median difference 0.00°, 90th
  percentile at most 1.5° in every speed band from 0 to over 10 cm/s), so the
  loader leaves `headings` empty; what `ha` holds is not stated. The `sp`
  column agrees with the speed computed from x, y as cm at 0.02 s per sample
  (r = 0.91), supporting both the cm and the period readings. The dandiset is
  a draft with no license stated.
- Harness workaround: `datasets/load_giocomo_000405.py` reads positions as cm and times from the period
- Proposed change: none for neurospatial
- Status: open

## G-034: 002030's ΔF/F scale is unclear

- Datasets: miniscope_002030
- Operation / step: loader
- Kind: data-quirk
- Evidence: the `Calcium` series is labelled dF/F, "stored exactly as they
  enter the analysis of the paper, negative values included". Values run from
  −16.9 to 979 (1st–99th percentile −13.4 to 171; per-cell medians −3.9 to
  13.1), larger than ΔF/F as a fraction; 0.7 % of its head-direction samples
  lie outside [−π, π], as the file says.
- Harness workaround: the ΔF/F event rule is relative to each trace's median and robust SD, so the scale does not change it
- Proposed change: none
- Status: open

## G-035: 001701's LFP conversion factor looks inverted

- Datasets: giocomo_001701
- Operation / step: loader (not used)
- Kind: data-quirk
- Evidence: `processing/probe_0_channel_160/LFP/LFP` is labelled `unit="volts"`
  with `conversion=1000000.0`; a raw value of 1 would be a megavolt. The
  survey does not use LFP.
- Harness workaround: none
- Proposed change: none
- Status: open

## G-036: 000405's pre-registered head-direction expectation assumed `hd` was head direction

- Datasets: giocomo_000405
- Operation / step: expectations
- Kind: data-quirk
- Evidence: "headings span the circle" was registered from the survey
  appendix, which read the `hd` column as head direction. It is the
  direction of movement (G-033), so the loader keeps no headings and the
  expectation fails ("check raised TypeError").
- Harness workaround: none (the expectation stays as registered)
- Proposed change: none
- Status: won't fix (the expectation, not the loader, was wrong)

## G-037: `detect_laps` finds 1 lap where a rat ran about 100 laps of a circular track

- Datasets: grosmark_000044_circular
- Operation / step: behavior.laps (`detect_laps(env, times, positions)`, defaults)
- Kind: wrong-result
- Evidence: Achilles 11012013 runs the 1 m circular track for 45 min. The
  angle of the 2-D position around the track's centre turns −97 times net
  (222 turns counting both directions), and the file's linearized coordinate
  wraps 123 times. On an environment built from the samples (a ring with a
  real hole), `detect_laps` returns 1 lap with its defaults; `min_overlap=0.5`
  returns 34; `max_gap=None` or `max_gap=10.0` returns 0. 18.5 % of samples
  are untracked (NaN) and the tracked samples have 23 gaps over 1 s. The
  default method matches laps against a reference lap at 80 % overlap; on a
  real trajectory with tracking gaps almost no lap reaches that.
- Harness workaround: none
- Proposed change: undecided; find which step loses the laps (reference-lap
  choice, overlap on gappy laps, segmentation at gaps) and test on this
  session, where the true count is known from the angle.
- Status: open
- More evidence (CRCNS batch): hc18_train261, a model train on a closed
  circuit. The unwrapped angle of the position around the circuit's centre
  gives 13.8–18.0 net laps per travel session, 107 over the seven (the
  documentation: about 15 per session). `detect_laps` with its defaults
  returns 1 clockwise lap. Tracking is 93–98 % complete in each session, so
  this is a loop with few gaps, and the defaults still miss nearly every lap.

## G-038: Replay on a track recorded in 2-D needs the track coordinate

- Datasets: grosmark_000044_circular, grosmark_000044_linear
- Operation / step: replay.compare_methods
- Kind: missing-feature
- Evidence: with 2-D positions, `compare_replay_methods` refuses: "needs a
  position along a track, but this is a 2-D RegularGrid environment", and
  suggests `Environment.linear_track(endpoints=...)`,
  `Environment.maze(...)` or `Environment.from_graph(...)`. The refusal is
  right and says what to do. For the 1.6 m linear track, a user would need
  the track's endpoints, which the file does not state. For the circular
  track no factory fits: it is a loop (G-021).
- Harness workaround: the operation decodes on the file's own linearized
  coordinate (`params["linearized"]`, a 1-D environment), status
  `workaround`. On the circular track that treats the loop as a line (G-021).
- Proposed change: the loop factory agreed for G-021; nothing for the linear
  track beyond the existing message.
- Status: open
- More evidence (NWB breadth batch): the 001695 novel linear maze and the
  000568 110 cm linear track hold ripples and 2-D positions with no
  linearized coordinate; the refusal is the same, and neither file states the
  track's endpoints in camera coordinates. On 001880's cheeseboard (an open
  maze, no track) see G-060.

## G-039: Pre-registered expectations that were wrong in the second batch

- Datasets: gardner_2022_grid, grosmark_000044_circular, grosmark_000044_linear
- Operation / step: expectations
- Kind: data-quirk
- Evidence: Gardner rat R day 1 has 483 module units, not the 250–450
  expected from memory of the paper. The 000044 loaders add the file's sleep
  states as epochs (as the figure-8 loader does), so the epoch names are not
  only `pre`/`maze`/`post`. The 000044 position extents are wider than
  expected, and the scale is not the cause: 2-D steps match the linearized
  coordinate's steps (median ratio 0.98 on the linear track, 1.01 on the
  circular one). On the 1.6 m linear track only 16 % of tracked maze-epoch
  samples are on the track (the linearized coordinate is defined); the rest
  lie off it, so the positions span about 2 m in each direction. On the
  circular track 72 % are on the track and the span is 134 cm, wider than the
  92 cm diameter a 290 cm circumference implies; why is not resolved.
- Harness workaround: none (the expectations stay as registered)
- Proposed change: none
- Status: won't fix (the expectations, not the loaders, were wrong)
- More evidence (environment fidelity): what the off-track 84 % of
  000044's linear-track samples are. The linearized coordinate covers the
  stem (0-160 cm, the paper's 1.6 m stem). Of the tracked samples, 61 % sit
  beyond its west end (x -51 to -26 cm, y -49 to 6 cm, about 1,000 s, slow:
  median 12 cm/s against 54 cm/s on the stem) and 17 % beyond its east end
  (x 129-137 cm): the two reward ends, where the head covers a wider area
  (25 by 55 cm at the west end) than the paper's 15 cm reward areas on the
  track; 3.5 % are the first 110 s, before the rat reached the track
  (y 64-161 cm); 2.2 % lie beside the stem. The environment from all samples
  has 1,096 bins in 21 parts (3 with 1 % of the bins or more: the track with
  its ends, 872 bins; where the rat was before the track, y 104-164 cm, 178;
  12 on the way between). On the circular track the
  linearized coordinate's two ends (0 and 290 cm) sit 27 cm apart in the
  camera, so the coordinate does not close the loop; a circle through the
  centreline is about 106 cm across, the paper's 1 m maze.

## G-040: Vollan files are speed-filtered and list fewer sessions than they hold

- Datasets: vollan_lt, vollan_of
- Operation / step: loader
- Kind: data-quirk
- Evidence: the authors removed every sample slower than 5 cm/s, so `t` has
  gaps (132 longer than 2 dt on the linear track, the longest 332 s). The
  documentation says so, and also gives `dt` and `nt`: a reader building
  times from them would close the gaps and shift every later sample against
  the spikes (hazard `times_rebuilt_from_dt`). The linear-track file's
  `sessions` lists 7227–7486 s, but samples run to 8412 s (55 % lie outside
  the listed session). Grid modules are defined on the open-field session:
  the linear-track file holds 84 of its module's 85 units.
- Harness workaround: none; gaps kept, every sample kept, module counts
  checked as at most the module size
- Proposed change: none for neurospatial (the hazard row shows whether it
  notices rebuilt times)
- Status: won't fix (data quirk)
- More evidence (archive batch): run on vollan_lt and vollan_of, `compute_boundary_vector_rates` warns that "about 6 % of the analysed time is in runs of removed samples lasting 0.25 s to max_gap=1.0 s, which looks like times were boolean-masked": correct, the authors removed every sample under 5 cm/s.

## G-041: 000044 has no ripple table; replay candidates are population bursts

- Datasets: grosmark_000044_circular, grosmark_000044_linear
- Operation / step: loader, replay.compare_methods
- Kind: data-quirk
- Evidence: a walk of the whole file finds units, position, epochs and a
  sleep-state table, but no ripple or candidate-event intervals.
- Harness workaround: the loader finds population bursts in POST non-REM
  sleep by a stated rule (multiunit rate in 1 ms bins, 15 ms Gaussian; peak
  above mean + 3 SD; bounded by mean crossings; 50–500 ms): 2010 bursts on
  the circular track, 1646 on the linear track; the first 200 are decoded.
- Result: on these candidates `DAVIDSON_2009` calls 0 of 200 replay on both
  sessions (95 % CI 0–1.8 %), `GROSMARK_2016` 9.5 % (circular) and 14 %
  (linear). The events are decoded on the linearized coordinate (G-038),
  sleep candidates are not ripples, and the circular track is cut into a line
  (G-021), so the difference is not yet a statement about the methods.
- Proposed change: none (neurospatial does not detect ripples from LFP;
  `replay.burst_onset_times` is per unit)
- Status: won't fix (data quirk)

## G-042: A speed column read as a position axis is flagged only by a memory warning

- Datasets: jadhav_000447
- Operation / step: io.nwb_readers, env.from_samples, env.occupancy, rates.spatial, behavior.segment_by_velocity
- Kind: silent-hazard
- Evidence: 000447 stores x, y and speed in one `SpatialSeries` under the
  unit "centimeters; centimeters/second". `read_position` refuses it with a
  clear message ("has 3 columns and declares unit 'centimeters;
  centimeters/second', which is not one length or pixel unit … a column may
  hold speed"), which is the right behavior and why the `io.nwb_readers` cell
  is an error. A user who reads the array with pynwb and passes all three
  columns gets a 3-D environment with only a memory warning (a
  48 × 44 × 118 grid), occupancy spread over speed (median 0.60 → 0.04 s per
  bin), rate maps from near-empty bins (median peak 10 → 100 Hz) and 535 →
  686 running segments. Nothing says the third axis is not a length.
- Harness workaround: the loader keeps columns 0 and 1 and checks that column
  2 tracks the speed from x, y (r = 0.71)
- Proposed change: the reader-coverage plan's column choice (proposal 6);
  nothing on raw arrays can tell speed from a third spatial axis
- Status: open

## G-043: On Vollan's speed-filtered data, real units carried less spatial information than their circular shifts

- Datasets: vollan_lt, vollan_of
- Operation / step: rates.information_vs_shifts
- Kind: harness-bug (fixed)
- Evidence: open field (rat 26018, 644 units, samples-built environment):
  median information 0.436 bits/spike, median over 4 shifts 0.596; 22 % of
  units beat every shift. Linear track: 0.289 vs 0.424, 12 %. The cause
  was the harness's shift rule: `operations.shifted_trains` shifted spikes
  circularly over the whole span `[times[0], times[-1]]`, while the maps
  count only the analysis epochs, the speed gate and tracking gaps (the
  recordings are speed-filtered, so most of the span is gaps). On the first
  150 open-field units 39 % of shifted spikes stayed inside counted time
  against 99.9 % of real ones; fewer spikes give higher bits per spike, so
  the null was inflated. Spikes are now shifted within the counted
  intervals (`analysis_settings["observed_intervals"]`, joined end to
  end). Rerun: open field 0.436 vs 0.238, 84 % beat every shift; linear
  track 0.289 vs 0.109, 86 % (declared environments: 0.380 vs 0.188, 84 %;
  0.152 vs 0.048, 79 %).
- Harness workaround: none needed after the fix
- Proposed change: none in neurospatial
- Status: resolved (harness fix)

## G-050: Position and head orientation stored one axis per series cannot be read

- Datasets: rsc_000987
- Operation / step: loader, io.nwb_readers (`read_position`, `read_head_direction`)
- Kind: missing-feature
- Evidence: 000987 stores motion-capture tracking as three `SpatialSeries`
  per container: `Position/x`, `y`, `z` (unit blank, reference frame blank)
  and `CompassDirection/rx`, `ry`, `rz` (radians, 0 to 2 pi). With default
  arguments `read_position` refuses ("Found 3 Position series candidates")
  and suggests `position_name='acquisition/Position/x'`, which returns one
  axis; there is no way to ask for (x, z) as one 2-D position, and nothing
  says which axis is vertical. `read_head_direction` refuses the same way for
  the three Euler angles. Only the values tell that y is vertical (span 9 cm
  against 53 cm for x and z) and that the blank unit is metres. The refusals
  are right; the cell is an error because a reader raised, as the operation
  defines it.
- Harness workaround: `datasets/load_rsc_000987.py` reads the five series with
  pynwb, returns (x, -z) x 100 cm and `ry` as heading
- Proposed change: accept a list of per-axis series (`position_name=[...]`)
  or document the pynwb recipe; say in the refusal when the candidates look
  like one axis each (same timestamps, one column each)
- Status: open

## G-051: Motion-capture head direction sits 0.3-0.5 rad off the direction of movement

- Datasets: rsc_000987, kanter_2025
- Operation / step: loader (`_record.heading_vs_movement`)
- Kind: data-quirk
- Evidence: both files track a rigid body on the head with motion capture.
  000987's yaw (`ry`, in the (x, -z) floor frame) against movement faster than
  10 cm/s at 10 Hz: mean offset +0.51 rad, spread 49 degrees (the mouse
  walks slowly: median 6 cm/s, 90th percentile 14 cm/s). Kanter's
  `head_direction` ("(x=1, y=0) unit vector") against movement faster than
  20 cm/s: mean -0.33 rad, spread 17 degrees (-0.42 rad, 34 degrees above
  10 cm/s). Neither passes the survey's frame rule (|mean| <= 0.2 rad and
  spread < 45 degrees). The offset is not fixed: over Kanter's 17 windows of
  60 s with at least 10 running samples (median 36), the window means range
  from -0.62 to +2.33 rad (circular SD 31 degrees), so no single correction
  is defensible from movement; its cause (a rigid-body zero, head turns while
  running) is not known. For contrast, 001701's tracked heading gives
  -0.07 rad and 10 degrees. The sign of each heading is tested by negating
  it alone: the spread rises to 87 degrees on 000987 (from 49) and to 81 on
  Kanter (from 17), so both turn the same way as their position frames.
  Mirroring positions and heading together leaves heading minus movement
  unchanged and cannot be tested this way (G-020).
- Harness workaround: none; both loaders keep the headings and record the
  frame as not verified in `conversions`. Mean vector lengths do not depend
  on the frame; preferred directions are not interpreted.
- Proposed change: none for neurospatial. The frame rule may need a
  per-dataset offset allowance for motion-capture heads: decide before using
  it as a hard gate.
- Status: open

## G-052: 001695's position conversion factor contradicts its values

- Datasets: gonzalez_001695
- Operation / step: loader, io.nwb_readers (`read_position`)
- Kind: data-quirk
- Evidence: `AnimalPosition/Position` is labelled `unit="centimeters"` with
  `conversion=0.01`. NWB defines the value as stored x conversion in the
  stated unit, so `read_position` (correctly, by the spec) returns a track
  0.066-1.14 "cm" long with no warning. The stored values span 108 by 6, a
  linear track in cm, and the file's own `Speed` (cm/s) matches the speed
  from the stored values (median ratio 0.90), not from the converted ones.
  The `io.nwb_readers` cell is a workaround: "position: mismatch: values
  differ". A user who takes the reader's output gets the hazard
  `conversion_applied_labelled_cm`: `Environment.from_samples(bin_size=2.5)`
  warns that the bin is at least the extent, and occupancy and rate maps
  raise "Interval filtering excluded ALL trajectory intervals ... check that
  min_speed=4.0 is in the positions' length units" (both caught), but
  `segment_by_velocity` silently finds 0 running segments instead of 315
  (G-014). The reference frame also says "(0,0) is center of arena" while
  every value is positive.
- Harness workaround: `datasets/load_gonzalez_001695.py` reads the stored
  values with h5py and checks them against the file's speed
- Proposed change: none for neurospatial (the reader follows the spec)
- Status: open

## G-053: 001754 samples position at spike times and codes a lost LED as a corner

- Datasets: neurolab_001754
- Operation / step: loader, env.occupancy, rates.spatial, decode.position
- Kind: data-quirk
- Evidence: the 25,320 position samples are exactly the 25,320 distinct spike
  times of the 72 units ("sampled at spike occurrence times"); within a
  session the median gap is 20-70 ms, the 90th percentile 0.27-0.5 s.
  Occupancy from these samples weights each position by the time to the next
  spike of any unit, not by time spent there. neurospatial's
  boolean-masking warning fires on every occupancy-based operation ("About
  17% of the analysed time is in runs of removed samples lasting 0.25 s to
  max_gap=1.0 s, which looks like times were boolean-masked"): it notices the
  irregular sampling, though its example cause (a speed mask) is not this
  one. Separately, 2,218 samples sit at exactly (0, 255), a corner no other
  sample reaches: the tracker's lost-LED code. Kept as positions (hazard
  `lost_led_code_kept`), they move 127 s of occupancy into one corner bin
  (most-occupied bin 34.6 -> 131.0 s) and lower the median peak rate from
  0.051 to 0.029 Hz, with no warning (the G-017 pattern, at a corner instead
  of the origin). Position decoding on the staircase is worse than chance
  (median error 103 px against 72 px).
- Harness workaround: the loader sets (0, 255) to NaN; occupancy is left as
  the sampling allows
- Proposed change: none beyond G-017's repeated-coordinate check, which would
  catch (0, 255) as well as (0, 0)
- Status: open
- More evidence (environment fidelity): the driver's environment pools the
  five sessions, four of which overlap in the image (the 1st-99th
  percentile of x spans about 0-255 px in sessions 2-5; session 1, a BL
  run, spans only x 16-40 by y 166-220 px): 2,174 bins, one main part. The staircase session alone gives 1,233 bins
  in 23 parts (2 with 1 % of the bins or more), 74 holes and 58 % of its
  bins holding fewer than 5 samples; its samples fill x 0-239 and y 81-254
  px (1st-99th percentile) with no visible track, so whether staircase parts at different
  heights overlap in the overhead view cannot be read from the data, and the
  layout (DANDI 001754: three yaw and three pitch turns of 90 degrees closing
  a loop) gives no dimensions to compare. Not verified.

## G-054: 000568's video timestamps run 22.6 s ahead of its position timestamps

- Datasets: valero_000568
- Operation / step: loader (hazard `video_timestamps`)
- Kind: data-quirk
- Evidence: the tracking video `ImageSeries` and the position series have the
  same 67,131 frames, but the video timestamps are 22.55-23.12 s earlier.
  The position timestamps are the ones aligned with the spikes: mean spatial
  information over the 35 units is 1.09 bits/spike with them, 0.81 with the
  video timestamps, and at most 0.85 at any shift of 5-40 s either way. With
  the video timestamps (hazard), median peak rate falls 1.57 -> 0.92 Hz and
  nothing warns: a shifted clock is indistinguishable from a real one in the
  arrays.
- Harness workaround: the loader uses the position timestamps and checks the
  offset
- Proposed change: none
- Status: open

## G-055: 000165 repeats timestamps and its spikes cover 270 s of 1,504 s tracked

- Datasets: aeryjones_000165
- Operation / step: loader, rates.spatial (hazards `duplicate_timestamps_kept`, `tracking_beyond_spikes`)
- Kind: data-quirk
- Evidence: the 45,114 position samples come exactly 30 per second overall
  (1,504 s), but only 32,340 timestamps are distinct: 12,774 intervals are
  zero (36 % of those pairs at the same position), 213 are under 1 ms,
  3,620 are 1-30 ms and the rest 30-104 ms, most near 52 ms. neurospatial refuses them
  clearly ("times must be strictly increasing; 12774 timestamp(s) repeat the
  one before ... at indices 1 and 2"), so the hazard is caught. The `units`
  table is a "MUA table" (one multi-unit train per channel, 32), every
  non-empty train starts with a spike at exactly 0.0 s, and spikes run
  continuously only to 269.6 s, then a 0.7 s fragment at 461.3-462.1 s; the
  LFP is 275 s long and tracking 1,504 s. neurospatial found the gap: on a
  first run analysed to the last spike it warned "All 32 units are silent
  from 269.6 s to 460.3 s (191 s) while position is tracked. If the
  electrophysiology was not recording then, pass spike_window=(start,
  stop)" (correct, and how the loader's analysis span was set). Analysed over
  the whole tracking (hazard `tracking_beyond_spikes`), rates are divided by
  1,504 s instead of 270 s of occupancy (median peak 4.58 -> 1.03 Hz) and the
  same warning fires for 462.1-1503.8 s (caught). The spike clock is ASSUMED
  to be the tracking clock with offset 0 (every channel has a spike at
  exactly 0.0 s, and the LFP starts at 0 too); the data do not show it: a
  scan of constant offsets (5 s steps, mean 1-D Skaggs information of the 29
  channels with at least 50 spikes) ranks offset 0 11th of 247, below
  offsets of 325-465 s, and multi-unit channels are too weakly tuned to name
  another. The length unit is labelled
  centimeters, but the tracked span is 35 by 5 and the 90th percentile speed
  6.6 per s, too short and slow to confirm cm for a mouse linear track.
- Harness workaround: the loader averages samples that share a timestamp,
  drops the 0.0 s spikes, limits analysis to 0-269.6 s and keeps positions in
  arbitrary units
- Proposed change: none for neurospatial (the duplicate-timestamp refusal and
  the silence warning already point at the problems)
- Status: open

## G-056: 001845's smallest file has 123 s of pose, repeated first timestamps and one unconverted keypoint

- Datasets: dyads_001845
- Operation / step: loader, io.read_pose, io.nwb_readers
- Kind: data-quirk
- Evidence: the DeepLabCut `PoseEstimation` (six keypoints, pixels) covers
  873.8-996.9 s of a 1,577 s video and session; the dandiset's usual files
  (about 9 MB, e.g. sub-ER1 ses-09) cover whole sessions. Its first nine
  timestamps are equal (873.812 s), so `read_pose` refuses: "Pose bodypart
  'PoseEstimationSeriesBody' timestamps must be finite and strictly
  increasing. Fix: correct the source time axis before reading." (right, and
  why the `io.read_pose` cell is an error). The raw keypoints have no NaN;
  low confidence is a separate array (head below 0.6 on 43 % of samples). The
  authors' processed positions (`processed_position/<keypoint>`, cm) drop the
  repeats and set low-confidence samples to NaN (head 38 %, nose 45 %, body
  32 %), but `processed_position/tail_end`, labelled centimeters, spans
  13-522 by 179-908, the pixel range: it was not converted. Chosen as
  positions (hazard `unconverted_keypoint`), it builds a 1,062-bin
  environment 730 "cm" wide instead of 245 bins and 125 cm, and 27 movement
  segments instead of 10, with no warning. With default arguments
  `read_position` refuses the six processed series ("Found 6 Position series
  candidates"), as for G-012.
- Harness workaround: `datasets/load_dyads_001845.py` reads
  `processed_position/head` (the keypoint the authors' well visits use) with
  `read_position(position_name=...)` and keeps its NaN
- Proposed change: none for neurospatial
- Status: open

## G-057: `read_units` refuses a units table whose ids repeat across probes

- Datasets: kanter_2025
- Operation / step: loader, io.nwb_readers (`read_units`)
- Kind: missing-feature
- Evidence: the Kanter file's `units` table has 380 rows but 235 distinct ids:
  each probe (LEC, CA1) numbers its clusters from 0, and the electrode group
  tells them apart. `read_units` refuses: "The units table repeats id(s) [0,
  1, 2, ...]". The refusal is right (ids must identify units), but no argument
  reads the table with a unit key built from another column, and the message
  does not say how to proceed. A user who keys spike trains by the table id
  instead (hazard `units_merged_by_id`) merges 145 pairs of units from two
  brain areas into 235 trains (median peak rate 4.61 -> 7.76 Hz, median mean
  vector length 0.106 -> 0.092), and nothing can warn: merged trains look
  like ordinary units. The file's head direction and position read fine
  (`read_position(columns=[0, 1])`; the default read returns a third, height
  column).
- Harness workaround: `datasets/load_kanter_2025.py` reads the table with
  pynwb and names units `<area>-<id>`
- Proposed change: name the electrode-group (or another column) route in the
  refusal, or accept a key such as `unit_key=("electrode_group", "id")`
- Status: open

## G-058: `units="pixel"` is warned about; only "pixels" and "px" are registered

- Datasets: neurolab_001754, senzai_000003
- Operation / step: env.from_samples, io.nwb_roundtrip
- Kind: false-alarm
- Evidence: `Environment.from_samples(..., units="pixel")` warns
  "Environment.units='pixel' is not in the standard registry ['cm', 'm', 'mm',
  'pixels', 'px']. Free-form strings are allowed but may not be interpreted
  by tools that consume `units`". The singular is the natural spelling (and
  the survey harness's); `read_position` itself normalizes a file's
  "pixels" to "px". In hazard probes this warning reappears whenever a probe
  rebuilds the environment, and is counted as the probe's "new message".
- Harness workaround: none (the harness keeps "pixel")
- Proposed change: accept "pixel" as an alias of "px", as "meters" is of "m"
- Status: open

## G-059: Pre-registered expectations that were wrong in the NWB breadth batch

- Datasets: aeryjones_000165, dyads_001845
- Operation / step: expectations
- Kind: data-quirk
- Evidence: 000165's "sampling rate 25-35 Hz" assumed one sample per video
  frame; after averaging samples that share a timestamp the median interval
  is 0.0518 s (19.3 Hz). Its "spikes cover tracking" was registered as a
  consistency requirement and fails because the spike recording stops at
  269.6 s of 1,504 s tracked (G-055). 001845's processed positions are at
  28.6 Hz (median interval 0.035 s), not 29-31 Hz. Not wrong, but worth
  knowing: 001695's track-length expectation (90-130 cm) was written after
  the stored values (6.6-114) had been read during the file choice, so it
  restates the metadata walk rather than predicting it; the other new
  expectations cite their metadata sources in the same way.
- Harness workaround: none (the expectations stay as registered)
- Proposed change: none
- Status: won't fix (the expectations, not the loaders, were wrong)

## G-060: No replay score for events in an open arena or maze recorded in 2-D

- Datasets: ca1ca3_001880 (and the linear tracks of gonzalez_001695 and valero_000568, logged under G-038)
- Operation / step: replay.compare_methods
- Kind: missing-feature
- Evidence: all three files hold ripples and 2-D positions and no
  linearized coordinate. `compare_replay_methods` refuses each: "needs a
  position along a track, but this is a 2-D RegularGrid environment that is
  not a linearized track ... Fix: use a 1-D ...". On the 001695 and 000568
  linear tracks the refusal is right and is G-038's case (the files state no
  track endpoints; 000568 states the track's length, 110 cm, but not where it
  lies in the camera frame). On 001880's cheeseboard maze, an open 2-D
  arena with 195 ripples on the maze, no 1-D track exists: the published
  replay methods in neurospatial (`DAVIDSON_2009`, `GROSMARK_2016`) score
  1-D sequences, and it has no 2-D replay score.
- Harness workaround: none (cells are `err`)
- Proposed change: undecided; a 2-D trajectory score is a new method and
  needs a published reference before it is added
- Status: open

## G-061: The default rate map passes 7 GB for one unit on a 98,000-bin grid, after only a memory warning

- Datasets: senzai_000003
- Operation / step: env.from_samples, rates.spatial (hazard: the "meters" label trusted)
- Kind: performance
- Evidence: 000003's positions are pixels labelled metres (G-062). A user who
  trusts the label and converts to cm (x 100) with a 4 cm bin gets
  `Environment.from_samples` warning "Creating large grid with shape (4181,
  2494) (10,427,414 bins). Estimated memory usage: 11097.9 MB" (0.7 s, peak
  0.9 GB, 97,993 active bins). `compute_spatial_rates` with
  `method="binned"` then takes 1.0 s and 1.1 GB for one unit, but the default
  `diffusion_kde` passed 7.4 GB within 25 s for one unit and was stopped; in
  the driver the same probe drove the machine into 10.7 GB of swap and was
  killed after 22 min. The environment's memory warning is the only signal,
  and its estimate (11 GB for the grid) does not describe what the rate map
  then allocates.
- Harness workaround: the probe is removed from `load_senzai_000003.py`;
  the numbers above are from running its steps by hand
- Proposed change: estimate the diffusion operator's memory from the active
  bin count and refuse (or fall back with a warning) above a limit, naming
  `method="binned"` or a larger `bin_size`
- Status: open

## G-062: 000003's positions are pixels labelled metres, outside a Position container

- Datasets: senzai_000003
- Operation / step: loader, io.nwb_readers
- Kind: data-quirk
- Evidence: `acquisition/position_sensor0` and `position_sensor1` are bare
  `SpatialSeries` (not in a `Position` container), labelled
  `unit="meters"`, reference frame "unknown", at 39.0625 Hz over the whole
  17,050 s recording. Their values run 117-499 (spans about 160 by 110): an
  arena 160 m wide if they were metres, so they are video pixels, and no
  calibration is given. The two sensors sit about 14 px apart, but the
  direction from one to the other does not follow movement (spread 66-83
  degrees), so no head direction is derived. The file has no epochs table
  for its "open exploration and theta maze"; only sleep states label the
  awake periods. `read_position` finds neither series (G-032).
- Harness workaround: the loader reads sensor 0 with pynwb and keeps pixels
- Proposed change: none for neurospatial beyond G-032
- Status: open

## G-070: Zhang 2022's deposit holds cell summaries, not tracking

- Datasets: zhang_2022_ebc (dropped)
- Operation / step: dataset selection
- Kind: data-quirk
- Evidence: Figshare 21647423 (`Zhang et al Figshare data.zip`, 197 MB) holds
  MATLAB code and three MATLAB `table` files, `All_BDcells.mat` (370 rows,
  48 cells, 14 rats), `All_HDcells.mat` (`mdata_2box` 189 rows,
  `mdata_4box` 587 rows) and `All_TDcells.mat`, read with `mat-io`. Each
  row has one cell's spike times and spike indices into a tracking stream,
  and the authors' rate maps, egocentric maps and scores; no column holds
  per-sample positions or head directions (checked on `All_BDcells` and
  `All_HDcells`; the paper's data statement says "Summarized datasets").
  Without the tracking, no occupancy or egocentric map can be computed, so
  the dataset is dropped. The egocentric operation therefore has no
  dataset whose authors classified egocentric boundary cells to compare with.
- Harness workaround: none (not loaded)
- Proposed change: none
- Status: won't fix (data quirk)

## G-071: Pre-registered expectations that were wrong in the archive batch

- Datasets: takigawa_7114733, j16_23020757
- Operation / step: expectations
- Kind: data-quirk
- Evidence: the Takigawa extent check took the two 2 m tracks as one span
  (180–240 cm), but the tracks lie side by side in the camera frame and
  together span 249 cm in y (each track's own principal-axis extent is 186
  and 194 cm, G-076). J16's `headings_span_the_circle` assumed the
  file's `head_orientation` would be used as head direction; the loader
  leaves it out (G-072), so the check has nothing to measure.
- Harness workaround: none (the expectations stay as registered)
- Proposed change: none
- Status: won't fix (the expectations, not the loaders, were wrong)

## G-072: J16's head orientation sits 13° off the direction of movement

- Datasets: j16_23020757
- Operation / step: loader (frame check), hd.tuning, rates.egocentric_boundary
- Kind: data-quirk
- Evidence: `head_orientation` (radians, [−π, π], 36 of 36 direction bins
  covered) minus the direction of movement above 10 cm/s has a circular
  mean of +13.1° and a median spread of 20.4°; the offset has the same sign
  on all nine corridors (+3.9° to +20.5°) and is the same against the
  file's own `head_velocity` (+13.4°). The frame check refuses a mean
  beyond 0.2 rad (11.5°), so the loader leaves `headings` empty and keeps
  the series in `params["head_orientation"]`. A 13° rotation is not a frame
  error (those are 90°, 180° or a mirror); it is either the headstage
  mounting or the rat's head posture, and the file cannot tell which. 21 of
  the 203 units have no spikes in this epoch.
- Harness workaround: none (headings not used)
- Proposed change: none for neurospatial. For the harness: decide whether
  the frame check should separate a frame error (|mean| near π/2 or π)
  from a small constant offset.
- Status: open (question for the maintainer)

## G-073: Agarwal's `headDirDeg` is not a head direction

- Datasets: agarwal_24757638
- Operation / step: loader
- Kind: data-quirk
- Evidence: in session rec02, 78 % of tracked samples of `headDirDeg` are
  exactly 0 and the rest sit on 45° steps (106 distinct values in 4.7
  million samples); heading minus direction of movement above 10 cm/s has
  a median spread of 89.4°, what an unrelated angle gives. Untracked
  samples are stored as `xMM = yMM = 0` (0.20 %). The record documents
  neither.
- Harness workaround: the loader leaves `headings` empty and sets
  `xPix = yPix = 0` samples to NaN
- Proposed change: none
- Status: won't fix (data quirk)

## G-074: The Grieves lattice files leave units and frames undocumented

- Datasets: grieves_2020_lattice
- Operation / step: loader
- Kind: data-quirk
- Evidence: `lattice_3d.pox/poy/poz` have no stated unit; the speed of the
  stored positions is 10.03 × the stored velocity `pov` (median), and
  `pov`'s median of 11.7 is a climbing rat's speed in cm/s, so positions
  are mm. The head direction `lattice.poh` is the azimuth in the overhead
  camera's 2-D frame: against `lattice.pox/poy` heading minus movement has
  mean −0.8° and spread 42°. The 3-D frame's x, y are that frame mirrored in
  y and shifted, sample by sample (`lattice_3d.pox = lattice.pox + 165.6`,
  `lattice_3d.poy = 1039.6 − lattice.poy` mm, RMS 3e-5 mm), not rotated, so
  `−poh` is the heading in the positions' frame (mean +0.8°, spread 42°).
  An earlier reading, that the frame was rotated and the azimuth not
  usable, compared `poh` itself with the 3-D frame (mean −113°). Each tetrode's cluster 0 field holds only `spike_times`
  (0.68–1.5 million spikes, 5.7 million in all), the unsorted remainder.
  The zip uses Deflate64, which Python's `zipfile` cannot inflate.
- Harness workaround: the loader divides by 10 with the speed-ratio check,
  returns `−poh` as the heading after checking the mirror and the heading
  against movement, leaves out cluster 0, and `_fetch` inflates Deflate64 with
  `inflate64` and checks the CRC-32
- Proposed change: none
- Status: won't fix (data quirk)
- More evidence (environment fidelity): the session file does document the
  lattice: `lattice_3d.lx/ly/lz` hold its 147 bars in the positions' frame
  (mm; 97 cm each, the paper's 0.97 m cube) and `config3.lattice_type` is
  "diagonal", the tilted lattice; the loader now returns them as
  `geometry["lattice_bars_cm"]` (G-111 uses them).

## G-075: Shipley's position header says 30 Hz; the camera ran at 30.05 Hz

- Datasets: shipley_30902972
- Operation / step: loader
- Kind: data-quirk
- Evidence: `trialData.pos` concatenates 12 trials' tracking (146,101
  samples) and states `sample_rate 30.0 hz`; sample index / 30 Hz ends
  4870.0 s, 8 s after the last spike (4862.1 s). The file maps each curated
  spike to its position sample (`tetrode(k).pos_sample`); a line per trial
  through those pairs gives 30.047–30.064 Hz and trial offsets within
  0.4 s, worst residual 41 ms. `pos.dir` equals `dir_disp`: with one LED it
  is the direction of displacement, not head direction. The Kilosort
  `spike_times.npy` / 30 kHz for the 71 `good` clusters equals the 1,201,058
  curated tetrode timestamps exactly.
- Harness workaround: the loader times the position samples with the
  per-trial fit
- Proposed change: none
- Status: won't fix (data quirk)

## G-076: Takigawa's two tracks have different pixel calibrations; two sessions are duplicates

- Datasets: takigawa_7114733
- Operation / step: loader
- Kind: data-quirk
- Evidence: in RAT5_SESS1 each track's `cm_per_pixel` (from its 2 m length)
  is 0.388 and 0.443; the session's `cm_per_pixel` 0.426 is their average,
  and with it track 1 reads about 10 % long and track 2 about 4 % short.
  With each track's own factor, the tracks' extents along their principal
  axes are 186.3 and 194.2 cm (the tracks are not straight, so less than the
  2 m path), and the two stay at least 32 cm apart, so no 4 cm bin holds
  both. In the record, `RAT1_SESS2_*` and `RAT4_SESS1_*` are byte-identical
  (same MD5 for both the clusters and the position files), so one of those
  two sessions is not what its name says.
- Harness workaround: the loader converts each track with its own factor and
  range-checks each track's principal-axis extent (170–205 cm); a first
  version used the session average
- Proposed change: none
- Status: fixed in the loader (scale); won't fix (the duplicate sessions are
  the record's)

## G-077: The water-maze file gives no length unit

- Datasets: watermaze_10248158
- Operation / step: loader, behavior.segment_by_velocity
- Kind: data-quirk
- Evidence: EthoVision's raw coordinates and the arena description (pool
  circle radius 95, platform radius 10) carry no unit or pixels-per-cm
  calibration, in the file, the Figshare record or the Rtrack export. The
  loader keeps `position_units="arbitrary"`, so no speed threshold can be
  stated and `behavior.segment_by_velocity` is n/a. Each trial's clock
  starts at 0 and the file holds no time between trials.
- Harness workaround: trials laid 200 s apart on one clock, with a NaN
  sample marking each trial's end
- Proposed change: none
- Status: won't fix (data quirk)

## G-078: The harness's decoding chance level used straight-line distance against a geodesic error

- Datasets: all with `decode.position`
- Operation / step: decode.position
- Kind: docs
- Evidence: `DecodingResult.error_against` measures along the bin graph by
  default (`metric="geodesic"`, documented); the harness computed its
  chance level with `np.linalg.norm`. On open fields the two differ by at
  most about 8 %, but on mazes they do not: on the Olafsdottir Z-track the
  median error is 115 cm (geodesic) against a Euclidean chance of 112 cm,
  which read as chance; the geodesic chance is 142 cm. The harness now
  computes chance with `decoding_error(..., metric="geodesic")`, shuffling
  the true positions within each connected part of the bin graph: a global
  shuffle on Takigawa's two separate tracks puts 51 % of pairs an infinite
  path apart (median inf); within each track the chance is 68.3 cm, and the
  share of cross-part pairs is reported as `chance_frac_disconnected`.
  Every geodesic chance differs from the old straight-line one when a
  recording is rerun, open fields included (quantized to bin centres, and
  8-connected paths up to about 8 % longer than the straight line). A user
  writing the same baseline by hand would make the same mistakes: nothing in
  the result says which metric its error used.
- Harness workaround: none (fixed in the harness; tested on a U-shaped track
  and on two separate boxes)
- Proposed change: show the metric next to the error in the decoding
  tutorial, and give a chance-level helper that uses the same metric
- Status: open

## G-079: The large-grid warning suggests a smaller bin than the one in use

- Datasets: grieves_2020_lattice (hazard probed once by hand, not by the driver)
- Operation / step: env.from_samples, rates.spatial
- Kind: error-message
- Evidence: with Grieves' mm positions taken as cm (×10) and `bin_size=4.0`,
  `Environment.from_samples` warns "Creating large grid with shape (388,
  429, 371) (61,753,692 bins). Estimated memory usage: 83863.5 MB … Fix:
  increase bin_size; … For example, use bin_size=2.0". The example is a
  fixed string (`src/neurospatial/layout/helpers/utils.py:412`), here half
  the bin size in use. These numbers come from a one-off manual run, not
  from a committed script: the environment built in 1.3 s (109,862 visited
  bins, 4.4 GB peak), and in a first driver run, before the hazard was taken
  out of `HAZARDS`, `compute_spatial_rates` on it ran for over an hour at
  18 GB before the run was stopped. The
  warning is the only signal of the unit error, which makes it the right
  place to name the likely cause.
- Harness workaround: the hazard is documented in the loader and left out
  of the driver's probes
- Proposed change: compute the example from the current `bin_size` and the
  grid shape (e.g. the bin size that keeps the grid under the threshold),
  and say that a grid this large often means positions in a smaller unit
  than stated
- Status: open

## G-080: Pixels taken for cm (a 2–4× scale error) are silent

- Datasets: olafsdottir_5566548, shipley_30902972, takigawa_7114733 (hazard `pixels_labelled_cm`)
- Operation / step: env.from_samples, env.occupancy, rates.spatial, behavior.segment_by_velocity
- Kind: silent-hazard
- Evidence: each loader converts pixels with a calibration the source
  states (390 and 274 pixels per metre; 0.388 and 0.443 cm per pixel on
  Takigawa's two tracks). With the pixels kept and called cm, nothing warns
  on any of the three: the Z-track's environment grows from 556 to 4281
  bins (extent 172 → 664 "cm"), the radial maze from 697 to 3488
  (177 → 480), the two 2 m tracks from 952 to 3997 (252 → 592); speeds
  2.3–3.9 × too high pass the speed gates (running occupancy 1266 → 1702 s,
  1350 → 1737 s, 1093 → 1510 s) and median peak rates rise 2–8 × from sparse
  bins (3.8 → 30.2 Hz, 16.7 → 60.1 Hz, 11.3 → 25.3 Hz). The fewer-than-5-bins check (decision 1) cannot see a
  scale error in this direction; only a stated apparatus size can.
- Harness workaround: none (each loader range-checks its span against the
  apparatus with `checked_span`)
- Proposed change: none on raw arrays. Where an environment is built from an
  arena outline (`from_polygon`, or an `arena=` with a stated size), samples
  far outside it could warn.
- Status: open

## G-081: A position clock a fraction of a percent slow shifts spikes against positions silently

- Datasets: shipley_30902972 (hazard `clock_from_header_rate`)
- Operation / step: env.occupancy, rates.spatial, behavior.segment_by_velocity
- Kind: silent-hazard
- Evidence: timing Shipley's positions as index / 30 Hz, the header's
  nominal rate, instead of the camera's measured 30.05 Hz puts positions up
  to 8 s late by the session's end. Occupancy barely changes (1349.5 s
  both ways) but the median peak rate falls from 16.7 to 10.7 Hz, and no
  warning fires: the spikes all still fall inside the (slightly longer)
  tracking span. On Vollan (G-040) closing gaps shortened the span and the
  drop warning caught it; a drift that lengthens the span is invisible to
  that check.
- Harness workaround: the loader times samples from the file's
  spike-to-sample map (G-075)
- Proposed change: none in neurospatial (no value check can see a slow
  clock); a loader-side check, as here, is the only defence
- Status: open

## G-082: No egocentric boundary cell test, and no speed gate on boundary maps

- Datasets: gardner_2022_grid, vollan_of, vollan_lt, giocomo_001701 (the
  operation run on the earlier recordings with head direction)
- Operation / step: rates.egocentric_boundary
- Kind: missing-feature
- Evidence: `compute_boundary_vector_rates(..., direction_frame="egocentric",
  boundary_source="analysis")` runs on every recording with head direction
  (an environment from samples declares no walls, so the edge of the
  visited bins stands for them; the default `"physical"` raises as for
  border scores, G-005). The share of units whose peak lies in the outermost
  ring (45–50 cm), where the docstring says untuned cells peak by chance
  (16–31 % of untuned simulated cells), is 10 % of 483 units in Gardner's
  1.5 m box, 48 % of 644 in Vollan's open field, 33 % of 320 on 001701's
  X maze and 44 % of 389 on Vollan's linear track (where "walls" are the
  track's edges). Median distance information is 0.02–0.05 bits. There is no
  public verdict for boundary-vector or egocentric boundary cells (no
  `is_*` predicate; there are ones for place, head-direction, object-vector,
  egocentric object-vector and spatial-view cells),
  so these maps cannot be screened. `compute_boundary_vector_rates` also has
  no `min_speed=`, unlike `compute_spatial_rates`; the operation runs on all
  samples. No new recording in this batch has usable head direction (G-072,
  G-073, G-074; one LED on the others), and Zhang 2022's authors'
  classification could not be compared (G-070).
- Harness workaround: none (the operation runs with `boundary_source="analysis"`)
- Proposed change: a shuffle verdict for boundary-vector and egocentric
  boundary maps with the same `criterion=` contract as the other
  predicates; `min_speed=` on the boundary-vector entry points
- Status: open
- More evidence (faithful arenas batch): with the walls declared, the
  outermost-ring share falls from 48 % to 5 % of Vollan's open-field units
  (median preferred distance 42.5 -> 7.5 cm, median distance information
  0.021 -> 0.068 bits) and from 38 % to 20 % on Kanter (37.5 cm), while
  Gardner, with no holes in its samples-built environment, barely moves
  (10 % -> 13 %, 12.5 cm both). The 48 % was the edges of unvisited holes
  standing in for walls. Kanter's share of units peaking on the left also
  moves, 0.85 -> 0.47.

## G-090: The same session in CRCNS hc-11 and DANDI 000044 differs only in its sleep scoring

- Datasets: hc11_achilles, grosmark_000044_linear
- Operation / step: loader (cross-check, `studies/hc11_vs_000044.py`)
- Kind: data-quirk
- Evidence: Achilles 10252013 from both archives: 137 units paired one to one
  in the same order, every spike time identical (8,414,607 spikes), cell types
  agree on all 137 (pyramidal/interneuron against excitatory/inhibitory), the
  80,762 position samples agree to 5e-8 s with identical 2-D and linearized
  positions and NaN patterns, and PRE/MAZE/POST bounds are identical. The
  sleep states differ: hc-11's `sessInfo.Epochs` scores five states (Wake 20
  intervals / 5698 s, Drowsy 72 / 9967 s, NREM 57 / 15488 s, Intermediate
  31 / 510 s, REM 28 / 2653 s); 000044's `states` table has three (Awake 62 /
  16747 s, Non-REM 60 / 15890 s, REM 22 / 2071 s), not a merge of hc-11's.
  So the same burst rule finds 1803 POST non-REM candidates on hc-11 and 1646
  on 000044. 000044's unit ids (1–272) are not hc-11's cluster ids (shank ×
  100 + cluster); the order is the same.
- Harness workaround: none
- Proposed change: none for neurospatial; a replay result that depends on
  sleep epochs depends on which archive's scoring was used
- Status: won't fix (data quirk)
- More evidence: on the driver's operations the two loaders agree to about
  1e-11 (the 5e-8 s timestamp differences), except `replay.compare_methods`:
  `GROSMARK_2016` calls 16 % of the first 200 candidates replay on hc-11 and
  14 % on 000044, because the candidates come from different non-REM epochs.

## G-091: Pre-registered expectations that were wrong in the CRCNS batch

- Datasets: hc3_ec013_669, hc28_hpa
- Operation / step: expectations
- Kind: data-quirk
- Evidence: hc-3's cell table lists 125 sorted cells for topdir `ec013.38`;
  only 72 fire in session ec013.669 (expected 100–125). The cells were
  sorted over all of the day's sessions (3 novel midSquare sessions, a wheel
  task, three linear-track sessions), and each `.clu` header counts every
  cluster of the day, so a cell silent in one session is expected (the hc-3
  description says so). The hc-28 stub said day 1 epoch 2 is a W-track run;
  that was read from the day-2 task file. Day 1 epoch 2 is a linear-track run
  (`lin`); the loader uses day 1's first W-track run, epoch 4.
- Harness workaround: none (the expectations stay as registered)
- Proposed change: none
- Status: won't fix (the expectations, not the loaders, were wrong)

## G-092: hc-28's LED angle sits about 15–20° off the path, beyond the heading frame check

- Datasets: hc28_hpa
- Operation / step: loader (heading frame check)
- Kind: data-quirk
- Evidence: on HPa day 1 epoch 4, the angle from the back LED to the front
  LED minus the direction of movement above 10 cm/s, with one-frame steps:
  circular mean −25.5°, median spread 51.0° (the other LED order 154.5°).
  `rawpos` is whole pixels, so one-frame directions are quantised and most of
  that spread, and most of the direction dependence first seen (−85° to +22°
  across 45° sectors), is the check's artefact (G-098). With movement over
  longer steps: 3 frames −23.3°/39.5°, 8 frames −18.8°/25.7°, 15 frames
  −14.4°/18.8°, 30 frames −12.7°/17.1°; sector means at 15 frames −30° to
  +3°. The authors' own `pos` `dir` against their smoothed x, y: −21.9°,
  28.0°. This looks like a mounting offset of about −15 to −20° plus a
  residual of about 30° that depends on the direction of movement. The LEDs
  are 15.7 cm apart (median); `pos` x, y follow one LED (`arg` `diode` 1),
  7.9 cm from the midpoint.
- Harness workaround: the loader keeps positions and drops head direction:
  the mean is outside the check's 0.2 rad at every lag, and a rotation fitted
  to this session's movement would be tuned to the data
- Proposed change: none for neurospatial
- Status: won't fix (data quirk)

## G-093: hc-31 stores position in inches and time in microseconds; its interneuron marker is lost

- Datasets: hc31_anm190422
- Operation / step: loader
- Kind: data-quirk
- Evidence: `posMazeLin`/`posMaze` give `unitsPerCm` 0.3937 and `px` 2.54:
  the coordinates are inches (the 48 m track spans 0–1896 units, 4815 cm
  after conversion); times and spike `timeStamps` are microseconds
  (`unitsPerSecond` 1e6); the velocity column is inches per microsecond
  (at most 5.8e-5). Nothing but those two scalars says so. The readme says
  interneurons and poor units have NaN in `ID` and gives the rule
  `all(ID <= 2, 2)` to exclude them, but `ID` is uint8 (entries: 154 zeros,
  35 ones, 189 twos), which has no NaN; MATLAB's `uint8(NaN)` is 0, so a
  marker that was set is now indistinguishable from "silent" (7 units are 0
  in every column), and all 63 units pass the rule. The other routes are
  thin: `spAll.width` is empty, the stored `rate` is at most 1.29 Hz for
  every unit (median 0.19), so none looks fast-spiking, and
  `clustInf.score` holds four quality strings per unit ('3+', '4-', …),
  not a cell type.
- Harness workaround: the loader divides by `unitsPerCm` and
  `unitsPerSecond`, checks 2-D steps against linear steps (ratio 1.000) and
  stored speed against step speed (0.90), and keeps every unit; hazards
  `inches_as_cm` and `times_in_microseconds` put the stored units back
- Proposed change: none for neurospatial
- Status: won't fix (data quirk)

## G-094: Two-LED frames with both LEDs at one point give heading 0 (east)

- Datasets: hc3_ec013_669, hc2_ec013_527
- Operation / step: loader; hd.tuning (hazard `coincident_leds_as_east`)
- Kind: silent-hazard
- Evidence: 5.1 % of tracked frames of hc-3 ec013.669 and 2.3 % of hc-2
  ec013.527 have identical coordinates for both LEDs (the tracker gave one
  blob to both). The back-to-front angle is `arctan2(0, 0)`, which NumPy
  returns as 0, so a reader that does not check gets a heading of exactly 0
  (east) on those frames. The .whl format also does not say which LED is in
  front or whether y points down the image; the movement check picks the
  order (mean 6.2° on hc-3, 0.4° on hc-2; the other order about ±180°).
- Harness workaround: `_formats.two_led_heading` returns NaN on those frames
- Proposed change: see the hazard rows in `results/warnings.md`
- Status: open

## G-095: `Environment` warns on `units="pixel"`, which neurospatial's own unit table accepts

- Datasets: hc3_ec013_669, hc2_ec013_527, hc18_train261
- Operation / step: env.from_samples, io.nwb_roundtrip
- Kind: false-alarm
- Evidence: `Environment.from_samples(..., units="pixel")` warns
  "Environment.units='pixel' is not in the standard registry ['cm', 'm', 'mm',
  'pixels', 'px'] …" on every environment of the three pixel recordings. The
  registry (`src/neurospatial/environment/_publication.py:90`) lists `px` and
  `pixels` only, while the unit alias table that the rest of the package
  uses (`src/neurospatial/_units.py:40`) maps `px`, `pixel` and `pixels` to
  `px`; the alias table also accepts `centimeters`, `meters` and others the
  registry warns on. The warning also makes the hazard probes count a "new"
  warning on `rates.spatial`: the clean run records it once, in
  `env.from_samples`, and reuses that environment, while a probe builds its
  own environment inside the probed operation, so the warning is new there
  (three `warned` rows whose real response was silent or a different
  warning).
- Harness workaround: none (the harness's position unit is "pixel")
- Proposed change: one unit table; the registry check accepts every alias
  that `_units` resolves
- Status: duplicate of G-058 (same warning, found independently on the CRCNS pixel recordings)

## G-096: A missing-sample marker other than NaN or 0 passes silently

- Datasets: hc3_ec013_669, hc2_ec013_527, hc18_train261 (hazards `whl_minus_one_kept`, `pos_minus_one_kept`)
- Operation / step: env.from_samples, env.occupancy, rates.spatial, behavior.segment_by_velocity, behavior.laps
- Kind: silent-hazard
- Evidence: Neuroscope `.whl` and `.pos` files mark untracked frames with
  `-1` (hc-2 description). Kept as a coordinate, every untracked frame sits
  at (-1, -1) px, outside the video image: hc-3 540 → 553 bins and 8 s of
  occupancy more in all (6.1 s in the (-1, -1) bin), median peak 2.99 → 3.47
  Hz; hc-18 69 s more in all, 68.8 s of it in the (-1, -1) bin, and
  1 → 0 laps; hc-2 3210 → 3224 bins. Nothing warns, and the proposed origin
  check (check 1) would not fire, since the marker is not 0. A rule for "many
  samples at one exact coordinate" was rejected earlier because the bat's
  perch holds 15 % of its samples.
- Harness workaround: the Neuroscope readers in `_formats.py` turn -1 into NaN
- Proposed change: none in neurospatial beyond documenting that missing
  samples must be NaN; a Neuroscope reader would own the marker
- Status: open

## G-097: Spike times on the wrong clock pass silently when the spikes that leave tracking are outside the analysis epochs

- Datasets: hc18_train261 (hazard `clock_assumed_20khz`)
- Operation / step: rates.spatial
- Kind: silent-hazard
- Evidence: hc-18's `.res` samples are on a 32,552.083 Hz clock (stated in the
  `.xml`); hc-2 and hc-3 use 20 kHz. Read at 20 kHz, every spike time is
  1.63 times too late. With `epochs=` the seven travel sessions, the median
  mean rate falls 0.78 → 0.27 Hz and the median peak 39 → 2.0 Hz with no
  warning: the spikes pushed past the end of tracking are outside the
  requested epochs, and since #265 the drop warning counts only spikes inside
  them; inside the epochs, the spikes are merely misplaced. A spike-time
  scaling that keeps most spikes inside the recording looks like a quiet
  session.
- Harness workaround: the loader reads the rate from the `.xml`
- Proposed change: none decided; a check that the last spike is near the end
  of tracking (here 32,500 s against 19,964 s) would see it, but sleep
  recordings legitimately run on after tracking
- Status: open

## G-098: The harness's heading frame check is biased by whole-pixel positions

- Datasets: hc28_hpa (also any loader with integer-pixel tracking)
- Operation / step: loader (`_record.heading_vs_movement`)
- Kind: friction (harness)
- Evidence: `heading_vs_movement` takes the direction of movement from one
  sample to the next. On hc-28's `rawpos` (whole pixels at 30 Hz, steps of a
  few pixels) those directions snap to a few angles, which widened the
  spread to 51° and produced a direction-dependent offset that is mostly the
  check's own: over 15 frames the spread is 18.8°. A simulated head running a
  60 px circle in whole pixels at 1.6 px per frame gives a one-step spread
  above 10° against under 3° over 10 steps
  (`tests/test_record.py::test_heading_vs_movement_lag_removes_pixel_quantisation`).
- Harness workaround: `heading_vs_movement(..., lag=)` (default 1, unchanged)
  measures movement over `lag` samples; the hc-28 loader reports lags 1 and 15
- Proposed change: use a lag of several samples for integer-pixel tracking in
  future loaders
- Status: fixed in the harness (option added); earlier loaders keep lag 1

## G-110: An environment from samples joins the labyrinth's corridors through the walls between them

- Datasets: rosenberg_labyrinth
- Operation / step: env.from_samples (`studies/environment_fidelity.py`)
- Kind: wrong-result
- Evidence: in the repository's layout (`NewMaze`) the corridors are one
  cell wide and neighbouring corridors abut, so nose samples on the two sides
  of a wall fall in neighbouring 0.25-cell bins. `Environment.from_samples`
  (2,780 bins, one component) fills almost the whole 15 x 15-cell square: for
  random sample pairs (seeded) the path through its bins is a median 0.66 x
  the distance along the authors' corridors (5th percentile 0.28). Nothing in
  the samples shows the walls: 0.05 % of links have no sample within a bin of
  their midpoint (2.4 % none within half a bin). That test cannot find a
  wall thinner than a bin at all: when every bin holds a sample a link's
  midpoint is at most about 1.4 bins from one (a U of two 10 cm corridors
  behind a 0.5-3 cm wall merges with no far link). Against the stated
  layout the links show it: 795 of the grid's links (8.1 %) join bins more
  than 3 x the link (plus both bins' offsets) apart along the corridors
  (`studies/environment_fidelity.py` `jump_links`). The track-graph
  environment the driver also builds (`from_graph`, 350 bins) follows the
  corridors.
- Harness workaround: the loader declares the graph (one chosen cell wide),
  so every operation uses it (`BATCH_FAITHFUL_TRACKS.md`)
- Proposed change: say in the `from_samples` docs that it cannot see walls
  thinner than a bin, and point mazes to `from_graph` or to declared walls
  (`arena=` with walls, which close the links they cross); in the harness,
  use the graph environment for the labyrinth
- Status: open

## G-111: A voxel environment of the climbing lattice has no bars; a bar graph is buildable but not offered

- Datasets: grieves_2020_lattice
- Operation / step: env.from_samples (`studies/environment_fidelity.py`)
- Kind: missing-feature
- Evidence: the file stores the tilted lattice's 147 bars (`lattice_3d
  lx/ly/lz`, 97 cm long, crossings 16.2 cm apart; the paper's Supplementary
  Methods: a 6 x 6 x 6 cubic lattice 0.97 m a side, rotated to stand on a
  vertex). The 4 cm voxel environment has 8,193 voxels in one component. Head
  positions are a median 6.0 cm from the nearest bar (95th percentile 10.4
  cm) and 17 % of them are farther from every bar than the centre of a
  cube face (8.1 cm), so the head does move through the space between bars,
  and only 0.7 % of voxel links have no sample within a bin of their
  midpoint. But the voxel graph has no notion of the bars the rat stood on:
  for random sample pairs (seeded) its path length is a median 0.77 x the distance
  along the bars (5th percentile 0.62); for 5,478 pairs of samples at the
  middle of parallel bars that bound one cube face (16.6 cm apart in a
  straight line; about two spacings, 32 cm, along the bars) it is a median
  18.2 cm against 31.1 cm along the bars, and below 0.75 x the bar distance
  for 83 % of them. `Environment.from_graph` with 3-D node
  positions builds the bar network (343 crossings, 882 bar segments: 4,410
  bins, one component) and, with `track_width=` twice the 99th-percentile
  offset (30.8 cm), places 99.0 % of head positions on it, but no
  constructor takes a set of bars, and the edge order must list every
  segment (G-112). Head movement is only weakly aligned with the bars (22 %
  of 0.2 s steps within 20 degrees of a bar axis, 18 % for random
  directions), so the voxel and bar distances bracket the head's travel.
- Harness workaround: none (the driver's 3-D operations use the voxels)
- Proposed change: a documented recipe, or a constructor, for climbing
  structures from their bars (segments in 3-D); state in the 3-D docs that a
  voxel environment from head positions measures the head's space, not the
  climbing paths
- Status: open
- More evidence (faithful arenas batch): the loader now declares the bars
  (`_geometry.lattice_graph`: 343 crossings, 882 segments, all kept;
  `track_width` 30.8 cm = 2 x the 99th-percentile head-to-bar distance, no
  source states how far the head reaches; the Supplementary Methods give
  the tubes as 10 mm across). Against the bars the declared environment's
  path is a median 0.95 x the distance along the bars (voxels: 0.77), and
  for the 5,478 pairs on parallel bars of one face it is 30.4 cm against
  31.1 along the bars (voxels: 18.2). It places 99.0 % of head positions;
  57 % of its 4,410 bins get occupancy. Spatial information rises (median
  0.86 against 0.78 bits/spike), decoding is at chance on both (97.0 cm
  against chance 97.0 on the bars, 82.7 against 76.8 on the voxels), and
  `detect_laps` now finds a lap (G-134). The voxel environment stays
  available with `--inferred-env`.

## G-112: `Environment.from_graph` silently drops graph edges missing from `edge_order`, but still counts positions on them as on the track

- Datasets: grieves_2020_lattice (a bar graph built for this study); any
  graph with a loop
- Operation / step: env.from_graph
- Kind: wrong-result
- Evidence: bins and links come only from `edge_order`
  (`src/neurospatial/layout/helpers/graph.py:142` loops over it;
  `track_pairs`, line 285, joins only its segments' ends), but the track
  footprint (`TrackArena`) is built from every graph edge
  (`src/neurospatial/environment/_factories.py:805-808`). With
  `edge_order=list(nx.dfs_edges(graph))`, the order the driver's `_graph_env`
  uses, the lattice's graph of 882 bar segments becomes 1,710 bins over 342
  segments, a spanning tree: the 540 that close its loops are left out, with
  no warning; listing every edge gives 4,410 bins. A 100 cm square loop shows
  the consequence (repro below): the depth-first order keeps 3 of its 4
  edges (60 bins against 80), with 0 warnings, and with `track_width=10`
  `contains` still accepts (0, 50) on the dropped edge, while `bin_at` puts
  (0, 10) and (0, 50) both in bin 0 at (2.5, 0) and `occupancy` counts them
  there. A loop always loses its closing edge this way. The docstring says
  only that `edge_order` "defines the 1D bin ordering". The driver's graphs
  so far are trees (labyrinth, J16, Hafting), so its results are unaffected.

  ```python
  import networkx as nx, numpy as np
  from neurospatial import Environment
  g = nx.Graph()
  for k, p in enumerate([(0, 0), (100, 0), (100, 100), (0, 100)]):
      g.add_node(k, pos=p)
  g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
  order = list(nx.dfs_edges(g, source=0))  # [(0, 1), (1, 2), (2, 3)]
  env = Environment.from_graph(g, order, bin_size=5.0, track_width=10.0)
  pts = np.array([[0.0, 10.0], [0.0, 50.0]])  # on the dropped edge (3, 0)
  env.n_bins, env.contains(pts), env.bin_at(pts)  # 60, [True True], [0 0]
  ```
- Harness workaround: the study lists every edge after the depth-first ones
- Proposed change: refuse an `edge_order` that leaves graph edges out (or
  build the footprint from `edge_order` only and warn); in the harness, order
  every edge (depth-first first)
- Status: open

## G-113: The driver builds one environment from every sample, merging apparatus that share a camera frame

- Datasets: jadhav_000447, neurolab_001754, senzai_000003, grosmark_000044_linear,
  ca1ca3_001880
- Operation / step: env.from_samples and every operation that uses it
- Kind: wrong-result
- Evidence: `operations._env` bins every tracked sample; `analysis_epochs`
  restricts occupancy, not bins. 000447's novel W-track (runs 1 and 3) and
  familiar W-track (run 2) are rotated 90° in the same camera frame (DANDI
  000447 describes the novel, familiar, novel sequence): their union is one
  1,465-bin block about 110 x 105 cm with path lengths a median 1.07 x the
  straight line, while each W alone (1,172 and 788 bins) gives
  medians of 1.19 and 2.53 (95th percentile up to 4.9), the detours
  of a maze. The loader sets no `analysis_epochs`, so 000447's rate maps also
  pool the two mazes. 001754's five sessions (three apparatus) overlap in the
  image: all sessions give 2,174 bins, the staircase session alone 1,233.
  000003 pools an open field and a maze (stated by the loader). On 000044's
  linear track the first 110 s, before the rat reached the track, add a path
  up to y = 161 cm (3.5 % of samples). 001880's start box between the
  cheeseboard sub-sessions is outside `analysis_epochs` (`onMaze`) but in
  the bins: about 62,000 off-maze samples, most of them the box's held
  runs of identical samples (G-158). The loader keeps them as the file
  stores them (an earlier version set held runs to NaN, leaving 3,010).
- Harness workaround: none
- Proposed change: in the harness, build one environment per apparatus from
  the samples of its epochs (a loader parameter such as
  `params["environment_epochs"]`), and give 000447 one per maze
- Status: open

## G-114: The Vollan open-field session is round, not the stated 150 cm square

- Datasets: vollan_of
- Operation / step: env.from_samples (`studies/environment_fidelity.py`)
- Kind: data-quirk
- Evidence: Vollan et al. 2025 Methods state a 150 x 150 cm square box for the
  open-field rats and two exceptions: a 150 cm circular arena in a dark room
  and a 1 m x 1 m square with a transparent floor.
  Rat 26018's samples (`of/26018_2.mat`) fill a disc: their convex hull is
  0.81 of its minimum rotated rectangle and 0.94 of its minimum enclosing
  circle (a square explored to its corners gives about 1 and 0.64; Kanter's
  1 m square gives 0.98 and 0.65), the hull spans 139.7 by 135.0 cm, and a 150
  cm square placed on the samples is 57 % covered by bins with a median
  13.5 cm between its wall and the nearest bin. The 1 m square is ruled out
  by the 1.4 m span. Either this is the circular session or the rat avoided
  the corners; the file does not say which arena it is.
- Harness workaround: none (the loader says "an open field about 1.5 m
  across")
- Proposed change: none in neurospatial; record the arena per session in the
  loader once the source says which
- Status: resolved in the harness: the file's session type is
  `open_field_light` (`Dsession.sessions.type`). Of the 18 other open-field
  files downloaded for this survey, 16 are `open_field` with square samples
  (hull 0.98-1.00 of its minimum rectangle, one at 0.92) and 2 are
  `open_field_darkness` with round ones (0.79). The paper's one circular
  arena (150 cm) held the dark-room session and "a succeeding recording
  session with room lights on (same arena)" (Extended Data Fig. 9k), so the
  loader declares a 150 cm circle centred at the origin (`mat_files.md`:
  positions "relative to arena center"). The paper names no rat or file, so
  this is an inference from the session type and the samples' shape. The
  samples' bounding circle is 144 cm across (residual -4.3 %) and its centre
  1.65 cm from the origin.

## G-115: A 1-D environment from a linearized coordinate with gaps between edges falls into one part per corridor

- Datasets: j16_23020757 (and any track linearized with `edge_spacing`)
- Operation / step: env.from_samples on `params["linearized"]` (the replay
  operation's environment for a 2-D track, G-038)
- Kind: friction
- Evidence: J16's `linear_position` lays the nine corridors end to end with
  15 cm gaps. `Environment.from_samples` on it at 4 cm gives 129 bins in 9
  components (gaps of 16 cm between them), so the centre junction is lost and
  every corridor is unreachable from the others (end-to-end path infinite).
  The track-graph environment (`from_graph` with the same graph) keeps the
  junction. The replay operation does not run on J16 (no ripples), so no
  result changed.
- Harness workaround: none
- Proposed change: in the harness, decode a 2-D track's replay on its graph
  environment when the loader gives one; in neurospatial, say in the
  `from_samples` docs that a linearized coordinate with edge gaps needs
  `from_graph`
- Status: open

## G-116: 001845's W-maze environment falls into three parts

- Datasets: dyads_001845
- Operation / step: env.from_samples
- Kind: data-quirk
- Evidence: the file's 123 s of head positions (38 % NaN, G-056) give 245
  bins in 4 components, 3 of them with 1 % of the bins or more: the
  middle and lower arms are not joined to the stem, so path lengths between
  arms are infinite. A W-maze is one connected track (DANDI 001845: paired
  W-mazes).
- Harness workaround: none
- Proposed change: none in neurospatial; this file is the smallest of the
  dandiset and too short for a maze environment
- Status: open

## G-117: Tracking jumps on the Z-track add a second diagonal of bins

- Datasets: olafsdottir_5566548
- Operation / step: env.from_samples
- Kind: data-quirk
- Evidence: 140 steps between consecutive 50 Hz samples are longer than 10
  cm (the longest 166 cm). The isolated samples they leave make a sparse
  second diagonal and a vertical streak beside the Z (figure), so the
  environment has 556 bins in 7 components (2 with 1 % of bins or more),
  27 % of its bins hold fewer than 5 samples (0.1 s), and most stray bins
  join the main part, so the environment holds off-track places.
  `from_samples(..., bin_count_threshold=4)` keeps 425 bins (410 in the
  largest part).
- Harness workaround: none
- Proposed change: in the loader, drop samples that jump faster than a rat
  runs (the cause, reflections or a second light, is not checked); in
  neurospatial, none beyond `bin_count_threshold`
- Status: open

## G-130: 000987's samples span 52 cm where the study's square has 80 cm walls

- Datasets: rsc_000987
- Operation / step: loader (declared environment)
- Kind: data-quirk
- Evidence: the file says only "square arena". The study's preprint
  (Mehrotra, Levenstein et al. 2023, bioRxiv 10.1101/2023.09.12.557442,
  Methods) builds the square from "removable walls (width: 80 cm, height:
  50 cm)" and gives the RSC mice a 70 cm circle as well. The samples' fitted
  rectangle (after trimming 0.1 % per tail) is 52.6 x 52.1 cm, 34 % short of
  80 cm, and the samples fill a square (convex hull 0.98 of its minimum
  rectangle, 0.64 of its minimum circle), so this is the square, not the
  circle. A mouse tracked on the head reaches within a few cm of a wall in
  the other open fields here (000582: 99.5 x 97.1 cm in a 1 m box), so the
  gap is not the head's distance from the wall. Either the walls overlap to
  make a smaller square, or the position unit (read as metres from the
  value ranges, unlabelled in the file) is off by about 1.5.
  `_geometry.declared_rectangle` refuses an 80 cm square (residual -34 %,
  limit 15 %), so the environment stays built from the samples.
- Harness workaround: none (environment inferred)
- Proposed change: none in neurospatial; read the published methods (Current
  Biology 2024, not available here) for the square's size
- Status: open

## G-131: With declared walls, border scores are -1 for most cells where the tracked point stays a few cm off the wall

- Datasets: vollan_of, hc2_ec013_527 (also gardner_2022_grid, kanter_2025, sargolini_000582)
- Operation / step: cells.border_score on a declared environment
- Kind: wrong-result
- Evidence: declaring the stated outline (`Environment.from_polygon`) lets
  `border_scores()` run (G-005), but the bins along the wall are the ones
  the tracked head never enters: the samples-built environment's outermost
  bins sit a median 4.3 cm (Vollan's 150 cm circle) and 3.8 px (hc-2's
  180 cm square) inside the declared wall
  (`results/studies/environment_fidelity.md`). `observed_wall_fraction` of
  the first unit's map is 0.014 on Vollan's open field and 0.049 on hc-2
  (0.47 Gardner, 0.42 Kanter, 0.50 000582). The coverage term counts an
  unvisited piece of wall as not covered (documented in `border_score`,
  unlike BNT/opexebo, which look up to 8 bins inward), so coverage is near 0
  and the score is -1 for 407 of 619 units on Vollan's open field and 27 of
  36 on hc-2 (60 of 483 on Gardner, 20 of 337 on Kanter, 1 of 6 on 000582).
  No unit reaches 0.5 on Vollan or hc-2. With the samples-built
  environment and `boundary_source="analysis"`, the opposite happens: the
  edges of unvisited holes count as walls and 515 of 574 Vollan units, 258
  of 299 Kanter units and all 34 hc-2 units score above 0.5.
- Harness workaround: none (the scores are reported as computed)
- Proposed change: in neurospatial, a documented way to score a wall the
  tracked point cannot reach (BNT's inward lookup, or a wall offset by the
  tracking point's distance from the body), and a warning from
  `border_scores` when `observed_wall_fraction` is small; in the harness,
  report `observed_wall_fraction` beside the scores
- Status: open

## G-132: A wrong length unit inside a declared environment collapses the positions into a few bins without a warning

- Datasets: sargolini_000582, gardner_2022_grid, giocomo_000405, kanter_2025, vollan_of (hazard probes)
- Operation / step: env.occupancy, rates.spatial on a declared environment
- Kind: silent-hazard
- Evidence: with the samples-built environment, metres labelled cm
  (hazard `length_unit_off_by_100`, positions x 0.01) are caught by
  `Environment.from_samples` ("bin_size=2.5 is at least the per-axis data
  extent") and by `compute_spatial_rates` ("bandwidth=5 spans the
  environment"). With the stated arena declared, neither runs: the
  environment is the right size and the positions shrink into its centre.
  Occupancy piles into 1 of 1601 bins on 000582 (600 s), 2 of 3601 on
  Gardner, 2 of 1601 on Kanter, 4 of 2828 on Vollan's open field, and
  `compute_spatial_rates` returns maps 99.9 % NaN, with no warning. Positions
  read as mm on 000405 (x 0.1) cover 1.3 % of the declared 1.5 x 1.3 m field,
  again silent. Where the shrunk positions leave the declared outline (the
  002030 circle sits at 10-105 cm), occupancy raises ("Interval filtering
  excluded ALL trajectory intervals ... check that positions use the
  environment's coordinates"), and positions 100 x too large raise on the
  bat's room. A declared environment makes a unit error visible (the
  samples cover a tiny share of a known arena) but nothing reports it.
- Harness workaround: none (`env.declared` reports `frac_samples_outside`,
  which stays 0 when the positions shrink inside)
- Proposed change: warn in `occupancy` and `compute_spatial_rates` when the
  samples span a small share of an environment with a declared outline (for
  example under a tenth of its extent on every axis), naming the units
- Status: open

## G-133: Egocentric boundary maps refuse a recording in pixels unless `distance_range` is given

- Datasets: hc2_ec013_527
- Operation / step: rates.egocentric_boundary
- Kind: friction
- Evidence: `compute_boundary_vector_rates(..., direction_frame="egocentric")`
  raises "distance_range is required because env.units='pixel' is not a
  known length unit ... pass distance_range=(0.0, max_distance) and
  distance_bin_size in the units of positions", with the declared and the
  samples-built environment alike. The refusal is right: the default 0-50 cm
  rings cannot be placed without a scale. The loader has one,
  `geometry["inferred_cm_per_pixel"]` (0.995), but the operation passes the
  defaults.
- Harness workaround: none (cell `err`)
- Proposed change: in the harness, pass `distance_range` and
  `distance_bin_size` converted with the loader's inferred scale for pixel
  recordings; none in neurospatial
- Status: open

## G-134: `detect_laps` reports a lap on the 3-D lattice's bar graph

- Datasets: grieves_2020_lattice
- Operation / step: behavior.laps on the declared bar-graph environment
- Kind: wrong-result
- Evidence: on the voxel environment `detect_laps(method="auto")` refuses:
  "cannot tell whether this 3-D environment has a loop". On the declared
  environment, `Environment.from_graph` on the lattice's 343 crossings and
  882 segments, it accepts the graph (every cube face is a cycle) and
  returns 1 lap of direction "unknown". A climbing lattice has no loop a lap
  goes round: its cycles are the faces of its cubes, as a junction clique
  was for the labyrinth (G-015). The cell is `ok`.
- Harness workaround: none
- Proposed change: refuse a graph whose node positions are 3-D, as for 3-D
  grids, or ask for the loop explicitly
- Status: open

## G-135: Fewer finite grid scores on a declared arena with unvisited bins, from the smoothed maps

- Datasets: kanter_2025 (also vollan_of, sargolini_000582)
- Operation / step: cells.grid_score on a declared environment
- Kind: wrong-result (cause partly traced)
- Evidence: Kanter's finite grid scores fall from 235 (samples-built
  environment) to 144 (declared 1 m box, 35 % of its bins unvisited, NaN);
  Vollan's from 611 to 593, 000582's from 6 to 5. The units that lose a
  score are not grid cells (Kanter: a median -0.37 and at most 0.38 on the
  samples-built environment; Vollan at most -0.02; Gardner's grid cells keep
  every score). A check on Kanter
  (`compute_spatial_rates` on four environments, default and binned rates)
  separates the causes:

  | Environment | Bins (NaN) | Finite, `diffusion_kde` | Finite, `binned` |
  | --- | --- | --- | --- |
  | declared box | 1601 (35 %) | 144 | 361 |
  | samples-built | 1049 (0 %) | 235 | 364 |
  | samples-built on the declared grid's alignment | 1053 (0 %) | 262 | 361 |
  | the same with holes filled | 1512 (30 %) | 176 | 361 |

  The grid alignment is not the cause (aligned with the declared grid, the
  samples-built environment keeps 262). With unsmoothed rates the counts
  agree (361-364). The drop comes with the default `diffusion_kde` maps on
  an environment that holds unvisited bins (declared, or holes filled): the
  smoothing domain then includes those bins. Why the autocorrelogram of
  those maps yields a NaN score (the score's result reports no failure; one
  000582 unit had 2 peaks found) was not traced.
- Harness workaround: none (scores reported as computed; behaviour unchanged)
- Proposed change: trace in neurospatial how `grid_score` handles a
  smoothed map with NaN bins inside the arena, and say in its docs when a
  score is NaN rather than low
- Status: open

## G-150: The track factories say a track without a width is located by nearest-track projection, but `bin_at` refuses it

- Datasets: every declared track whose source states no width (figure8_000552,
  grosmark_000044_linear, grosmark_000044_circular, vollan_lt,
  aeryjones_000165, gonzalez_001695, hafting_2008_nird, hc31_anm190422,
  hc18_train261, takigawa_7114733, agarwal_24757638, giocomo_001701)
- Operation / step: env.declared, env.occupancy and every operation that
  places positions on a declared track
- Kind: docs
- Evidence: `Environment.from_graph` documents `track_width` as optional:
  "When omitted, the physical footprint stays unknown and the existing
  nearest-track lookup remains available"
  (`src/neurospatial/environment/core.py:4024`); `linear_track` and `maze`
  say "bin_at() remains nearest-track projection when width is unknown"
  (lines 3742 and 3898). On the declared figure-8 built without a width,
  `env.bin_at(positions)` raises `NotImplementedError: Positions cannot be
  assigned to the bins of a track built without track_width`
  (`geometry/_lookup.py:554`, `require_track_width`), and so do
  `occupancy`, `detect_laps` and `graph_turn_sequence`. The refusal is the
  safer behaviour and its message says what to do (G-016 records it); the
  three docstrings promise the opposite.
- Harness workaround: every declared track is built with a placement
  tolerance measured from the samples (`_tracks.declared_geometry`: twice
  the samples' 99th percentile distance from the graph, at least the
  physical width and one bin), recorded in `conversions` as a tolerance, not
  a width (G-152)
- Proposed change: correct the three docstrings: without `track_width` the
  track has bins and geometry but positions cannot be placed on it; say how
  to measure a width when none is stated
- Status: open

## G-151: A sample outside a track's width splits `detect_laps`' recording, so a narrow track loses most laps

- Datasets: figure8_000552, hc18_train261, grosmark_000044_circular
- Operation / step: behavior.laps (`detect_laps(..., method="region")`)
- Kind: wrong-result
- Evidence: the declared figure-8 (fitted graph, start region across the
  centre bar), 21 trials in the file. `detect_laps(method="region")` with the
  maze epoch finds, by track width (share of tracked samples outside it):
  6.6 cm (5.0 %) 2 laps; 10 cm (1.5 %) 14; 15 cm (0.7 %) 18; 30 cm (0 %) 20,
  alternating clockwise and counter-clockwise, as the task does; `max_gap`
  None changes none of these. The Notes say why: "A position outside the
  environment is observed but not located, so an interval with a sample
  there is not observed and splits the recording", and no lap spans a split.
  Head tracking leaves the footprint of a narrow track for a frame or two
  routinely, so a width that is the apparatus's (or a 95th percentile)
  removes most laps without a warning. hc-18's train circuit (declared loop,
  travel epochs, about 107 laps by the unwrapped angle, G-037): at the
  measured 35 px width (1.0 % of travel samples outside) 31 laps, 33 with
  `max_gap=None`; at 60 px (0 % outside) 92 and 97; at 120 px 93 and 99.
- Harness workaround: the placement tolerance is twice the 99th percentile
  sample distance (1 % outside by construction), not the 95th
- Proposed change: treat a position outside the environment like a missing
  one, bridged up to `max_gap`, or warn with the share of the recording lost
  to splits
- Status: open

## G-152: One `track_width` is both the track's physical width and the placement tolerance, and head tracking leaves the physical width

- Datasets: j16_23020757, hc28_hpa, hc3_ec013_669, jadhav_000447_familiar,
  valero_000568, shipley_30902972, rosenberg_labyrinth, jadhav_000447,
  olafsdottir_5566548, dyads_001845
- Operation / step: env.declared and every operation on a declared track
- Kind: missing-feature
- What `track_width` controls (read in `src/`, checked on a two-segment
  track at widths 6 and 30 cm): only which positions are placed
  (`TrackArena.contains`, and so `bin_at`, `occupancy` and every operation
  that bins positions) and the footprint (`TrackArena` capsules, used by
  `combine`'s component lookup and saved by the codecs). The bins, their
  measure (lengths), connectivity, smoothing kernels, and the occupancy and
  rates of the samples it places are the same at both widths.
- Evidence: the share of tracked samples farther than half the stated (or,
  for the labyrinth, chosen) physical width from the declared track,
  `environment_fit["outside_physical_width"]`, split by speed
  (stopped < `min_speed` <= moving < 30 cm/s <= fast): J16 63.8 % (6.5 cm;
  87 / 64 / 35 %), hc-28 37.3 % (7 cm; 24 / 38 / 57 %), hc-3 26.3 % (7 cm;
  19 / 34 / 15 %), 000447's familiar W-track 13.0 % (7 cm; 2 / 32 / 26 %),
  000568 12.4 % (6.35 cm; 12 / 14 / 9 %), Shipley 0.7 % (10 cm arms; 1 / 0
  / 1 %), the labyrinth 4.6 % (one cell; 17.9 % at the paper's 0.75 cell);
  on the three refused layouts (G-153), 000447's novel W-track 35.8 % (7 cm;
  22 / 50 / 44 %), the Z-track 7.6 % (10 cm; 30 / 9 / 7 %) and 001845 6.1 %
  (7 cm; 0 / 16 / 16 %). The pattern differs: on J16 and the Z-track the
  head is off the track most while stopped, on the W-tracks and hc-3 while
  moving, which fits swings at the turns more than posture at the wells. None of
  these is misplacement: the worst straight stretch of each track has its
  samples a median 0.03-1.81 (units of each file) to one side, under one
  bin. The tracked point is a head-mounted LED (or keypoint) above an
  unwalled track: the head swings past the edge, and a 2-D camera projects
  a raised head outward.
- J16 in detail (the authors' graph, 6.5 cm hallways): on each corridor the
  samples' median signed lateral offset is -1.3 to +1.8 cm, so the graph is
  placed right; the "bias" earlier read as about 3 cm was the median
  |offset| (4.2 cm), not a bias. Only 13.1 % of samples are more than 10 cm
  off; on every corridor the 95th percentile |offset| is 15-18 cm. It is
  not parallax: a uniform scale about a fitted centre comes out at 1.024 and
  leaves 57.2 % outside (63.7 % before) with worse per-edge offsets, and the
  best single translation ((-1.25, +2.00) cm) leaves 50.4 %; with another
  fit (truncated squared distance, (-1.46, +1.54) cm) the median distance
  falls only from 4.2 to 3.4 cm and two corridors' offsets grow to 2.6-2.7
  cm. It depends on speed: below 4 cm/s 89.5 % are beyond the half-width
  (4.2 % more than 10 cm off), 4-10 cm/s 65.5 %, 10-30 cm/s 62.7 % (32.0 %
  more than 10 cm off), above 30 cm/s 33.6 %; while running, 61 % of
  samples in the junction-end tenth of each edge are outside, 34 % mid-edge.
  This is the tracked head moving about a correct graph, through stopped
  head posture and swings at turns, not the reward ports (an earlier
  reading). No shift or scale is applied.
- Harness workaround: every declared track is built with a placement
  tolerance measured from the samples (twice their 99th percentile distance
  from the track, at least the physical width and a bin, so it places 99 %
  of them by construction), and the stated
  physical width is recorded apart (`geometry["declared_physical_width"]`)
  with the share beyond it reported as a tracking measure, not judged
- Proposed change: separate the apparatus's width from how far off it a
  position may be placed (a `reach=` or tolerance on the track, the
  footprint keeping the physical width), so a declared track can state its
  floor and still place head-tracked positions; J16 is the case for it
- Status: open

## G-153: Several mazes' fitted layouts disagree with the stated dimensions

- Datasets: olafsdottir_5566548, jadhav_000447, dyads_001845, hc28_hpa,
  jadhav_000447_familiar, shipley_30902972
- Operation / step: the loaders' declared environments
- Kind: data-quirk
- Evidence: the layout fitted to the samples (node positions; topology and
  width from the source) against the source's dimensions, the base and the
  mean arm of a W against its stated 80 cm footprint, each section of the
  Z, each radial arm's reach (`_tracks.declared_geometry`, refused beyond
  `_geometry.MAX_SIZE_RESIDUAL`, 15 %). The Z-track's sections are 165, 204
  and 159 cm against the stated 190, 220 and 190 (Olafsdottir et al. 2016,
  Online Methods; -13 %, -7 % and -16 %, not one scale), so it is refused:
  either the file's `pixels_per_metre` 390 is off or the stated lengths are
  not centreline lengths; neither can be checked here. 000447's novel
  W-track has a 93.5 cm base and arms of 87.1-105.1 cm (mean +19 %; the
  longest is 1.31 x 80 cm), against Tang et al. 2023's "~80 x 80 cm", and is
  refused; its familiar W-track, described with the same size, has a 79.0
  cm base and arms of 80.7-89.4 cm (+6 %) and is kept, so the two mazes are
  not the same size in this file's frame. hc-28's W-track has a 71.6 cm base
  and arms of 64.9-75.8 cm (-11 %, kept). 001845's W-maze (base 91.6 cm,
  arms 80.4-93.9 cm) is refused for its thinly sampled base (80 % of a base
  segment has no sample within a bin in 123 s of pose, G-116), not its size.
  With support judged per straight stretch of 20 bins (review round), the
  thin part is a short base segment between junctions, reported, and the
  W-maze is kept (base +14 %, arms +7 %; fidelity: matches).
  Shipley's radial maze agrees: centre to arm end 86.9-95.5 cm against the
  stated 80 cm arm plus a 12.35 cm platform radius (92.35 cm; -6 % to +3 %).
- Harness workaround: a refused layout stays inferred; a kept one uses the
  fitted lengths, which are the samples' frame, and states the comparison in
  `conversions`
- Proposed change: none in neurospatial
- Status: open

## G-154: 000044's circular linearized coordinate leaves out the reward area

- Datasets: grosmark_000044_circular
- Operation / step: params['linearized'] (G-039, G-021)
- Kind: data-quirk
- Evidence: a robust circle fit to the samples with a linearized coordinate
  gives a centre (47.3, 50.5) cm and a radius of 49.9 cm, the paper's "1 m in
  diameter" circular maze (Grosmark & Buzsaki 2016, Supplementary
  Materials; 314 cm round). The coordinate runs 0-290 cm and its ends are 27
  cm apart in the camera. The 28.3 % of tracked samples without a
  coordinate are on the track too (their radius 40-58 cm, 5th-95th
  percentile). The coordinate starts at -1.5 degrees (east of the centre)
  and increases counter-clockwise to 290 cm at -30 degrees; in the 30 degree
  sector between its end and its zero 90-97 % of samples have no coordinate
  (23,000 samples, 27 % of all tracked ones): about 26 cm of arc where the rat
  spends much of its time, matching the paper's "predetermined 30 cm reward
  area". So the file's
  "on track" share (72 %, G-039) is the share outside the reward area, and
  the 290 cm coordinate is the loop minus that area; the 134 cm extent of the
  samples is the head leaning off a 1 m circle.
- Harness workaround: the declared environment is the stated 1 m circle
  centred on the samples, a closed loop that includes the reward area
- Proposed change: none
- Status: open

## G-155: Occupancy and rate maps drop positions outside a declared track without a warning

- Datasets: takigawa_7114733, shipley_30902972, hafting_2008_nird,
  grosmark_000044_circular, j16_23020757 (hazard probes and the first run of
  this batch)
- Operation / step: env.occupancy, rates.spatial, behavior.laps
- Kind: silent-hazard
- Evidence: `env.occupancy` and `compute_spatial_rates` leave out every
  position the declared environment cannot place, with no warning. With
  J16's track built at the stated 6.5 cm (this batch's first run) that was
  63.8 % of the samples, occupancy 423.6 s against 802 s on the grid; at
  the placement tolerance the loaders now use, 1 % (792 s). Only
  `decoding_error` says that "time bins have a decoded or actual position
  outside the environment", and `detect_laps` warns when most positions are
  outside ("720604 of 776429 positions with finite coordinates (93%) are
  outside the environment" on hc-18, whose default count spans the
  sleep-box sessions). The hazard probes show the same silence when
  positions are wrong: pixels read as cm (`pixels_labelled_cm`) leave 5.96
  of 1,069 s on Takigawa's tracks and 33 of 1,344 s on Shipley's maze,
  Hafting's track with positions / 100 keeps 0.24 of 650 s, and on 000044's
  circle untracked samples moved to (0, 0) fall outside the loop and cut its
  laps by region from 65 to 15 (G-151), each with no warning of its own. An
  environment built from the samples cannot show this (every sample has a
  bin by construction); a declared one can, which makes the share outside
  the first check of a declared apparatus.
- Harness workaround: `env.declared` reports `frac_samples_outside`, and the
  tracks are built with a placement tolerance (G-152)
- Proposed change: give `occupancy` and `compute_spatial_rates` the warning
  `detect_laps` has, at a lower share (a few percent of the analysed
  samples), naming the share and the fix (the frame, the units or the track
  width)
- Status: open

## G-156: The fidelity study's `jump_links` flags links inside an open hub, where the arms' corridors overlap

- Datasets: none in this batch (no radial maze has an independent layout);
  a synthetic 8-arm star (the probe below)
- Operation / step: studies/environment_fidelity.py `jump_links`
- Kind: harness-limitation
- Evidence: `jump_links` projects each bin centre onto the nearest segment
  of the stated layout and calls a link a jump when the distance along the
  layout between its two bins is more than 3 x the link plus both bins'
  offsets. At a hub the arms' corridors overlap (8 arms 10 cm wide overlap
  out to 5 / tan(22.5 deg) = 12 cm from the centre), so two neighbouring
  bins there project onto different arms and are a detour through the hub
  apart along the layout. On 8 arms of 70 x 10 cm sampled uniformly, with
  `Environment.from_samples` at 2.5 cm, 128 of 3,504 links are flagged, all
  within 17.4 cm of the centre (72 within 12 cm); 6 arms give 35 of 2,679;
  a sampled 8 cm platform at the hub changes nothing (128). None joins
  corridors through a wall. The probe: arms at angles 2 pi k / n from the
  origin, samples uniform 0-70 cm along and +/-5 cm across each, the
  reference a star of n segments from (0, 0), counting `jump_links` and the
  radius of each flagged link's midpoint.
- Harness workaround: none; only references with an open hub are affected
  and none of this batch's judged references has one (the labyrinth's and
  Takigawa's are corridors; on J16's graph, whose junctions join three
  edges, no link is flagged). A radial or star maze judged against an independent
  layout would need the hub drawn as a platform polygon, or links inside
  the overlap of two corridors excused, which needs the corridor width.
- Proposed change: harness only (not neurospatial); give `Reference` an
  optional corridor width and excuse links whose bins both lie within half
  of it of a shared node
- Status: open

## G-157: Wall and segment support cannot tell a tracked point kept off the apparatus from apparatus drawn where the animal never went

- Datasets: giocomo_000405, vollan_of, hc2_ec013_527, grieves_2020_lattice
- Operation / step: `_geometry.judge_declared` (fidelity study, declared
  environments)
- Kind: harness-limitation
- Evidence: the support check flagged an outline side (a run of edges
  turning less than 10 degrees; a circle is one side) with no sample within
  3 bins over more than 15 % of its length, and a lattice bar segment with
  no sample within one bin over more than 25 % of it. Four declared
  environments failed only this check. On 000405 69 % of one side of the
  150 x 130 cm field has no sample within 7.5 cm; the samples' own
  rectangle is 8.2 % and 8.0 % short of the stated sides and their
  outermost 5 % stay 7.3 cm inside the nearest wall all round. On Vollan's
  150 cm circle 31 % of the wall has none within 7.5 cm (samples' diameter
  4.3 % short, centre 2.2 % of the radius from the stated one). On hc-2's
  elevated 180 cm square 36 % of one side has none within 3 bins (the
  samples' sides 2.1 % and 3.0 % short). On the lattice 90 % of the 882
  bar segments have over 25 % of their length with no head sample within
  one 4 cm bin: the head is a median 6.0 cm from the nearest bar. In each
  case the tracked point (a head marker) keeps off the walls or bars, and
  the same numbers would follow from a wall drawn a few cm too far out.
- Harness workaround: the maintainer's rule (2026-11-03): head samples fall
  short of walls and bars because the tracked LEDs sit on a headstage above
  the animal, so for declared outlines (arenas, circles, rooms) and the
  lattice the support share is reported (`fit["max_side_unsupported"]`,
  `fit["max_unsupported"]`), not judged. A misplaced or wrongly sized
  outline is caught by samples reaching past the opposite wall, the samples'
  fitted size against the stated size, and the stated centre. Track segment
  coverage, within the placement tolerance, stays a criterion. With it the
  four now match on their stated sizes (000405 -8.2 % and -8.0 %, Vollan's
  diameter -4.3 %, hc-2 -2.1 % and -3.0 %, the lattice's side +0.0 % of the
  stated 97 cm with the authors' bars), and all ten declared arenas and
  3-D environments match.
- More evidence (review round): the stated shapes are now placed after a
  radial trim of the samples, and the residuals are 000405 -6.9 % and
  -6.3 %, Vollan's diameter -4.0 %, hc-2 +0.0 % and -0.8 %. A room is now
  judged per face as an outline per side, and the bat's room is a
  mismatch (G-025), so nine of the ten match. Track support is judged per
  straight stretch of 20 bins, not per segment, with the 3 bins nearest a
  free end reported, as for walls; the lattice's segments (16 cm between
  crossings, below 20 bins) have their support reported. Its placement is
  now judged as a whole, horizontally (`_tracks.frame_offset`; the vertical,
  raised by a head above the bars, is reported), and it is a mismatch at
  9.4 cm (G-159), so eight of the ten match.
- Proposed change: none in neurospatial; in the harness, measure the
  remaining checks' sensitivity on planted misfits (phase 6, rungs 2-3)
- Status: resolved (harness rule)

## G-158: Runs of identical positions may be lost tracking or stillness, and nothing in the file tells them apart

- Datasets: ca1ca3_001880, valero_000568, rsc_000987
- Operation / step: loader (`_record.find_suspect_tracking`, a load warning)
- Kind: data-quirk
- Evidence: runs of 10 or more frames identical in every coordinate, on
  positions with sub-pixel values (about 64,000 to 154,000 distinct values
  per axis, none on integer coordinates): 001880 29 % of the frames, in 90
  runs up to 82 s (most in the start box between sub-sessions, G-113);
  000568 46 runs up to 6 s; 000987 0.4 % of the frames in 22 runs up to
  1 s. A tracker that loses the animal can repeat its last fix; a still
  animal at a well or in the box can repeat a value too where the estimate
  is coarse. The 100 cm/s limit for a fast sample has no citation: it was
  chosen as faster than a mouse runs (000568's 99.9th percentile step is
  218 cm/s, from jumps after held frames). No independent signal of lost
  tracking (an LED flag, a second LED, a quality field) was found in these
  files, and the runs' place, the speed before them and the step after them
  were not measured.
- Harness workaround: none: the loaders keep positions as stored and warn
  with the counts (the maintainer's direction, 2026-11-03: warn users of
  issues, do not change the data). An earlier version set held runs and
  fast samples to NaN.
- Proposed change: none in neurospatial; a reader of these recordings
  checks the runs before analysing stops (wells, ripples, replay)
- Status: open

## G-159: The lattice's head samples sit 9 cm horizontally off the authors' bars

- Datasets: grieves_2020_lattice
- Operation / step: `_geometry.judge_declared` (fidelity study, the whole
  lattice's frame offset, `_tracks.frame_offset`)
- Kind: data-quirk (unconfirmed)
- Evidence: one translation fitted to the median offset, perpendicular to
  each bar, of the head samples near each of 738 bar segments (each segment
  weighted once, refitted until it settles: 22 steps) is
  (-8.2, +4.6, +2.2) cm, 9.4 cm (2.35 bins of 4 cm) horizontally. The fit
  is well posed (eigenvalues of the summed projections 482-497: the bars
  run in three directions). Independently, the shift of the samples on a
  2 cm grid that brings a seeded 20,000-sample subsample closest to the
  bars is (-8, +4 to +6, +2) cm, and it lowers the median distance to the
  nearest bar from 6.0 to 3.4 cm. The samples' 0.5-99.5 % extent is offset
  the same way in y (centre +5.1 cm from the bars' centre; x -2.9 cm). A
  head on a headstage above the bars raises z, which is reported, not
  judged; a head held ahead of the body turns with the animal, so it would
  not give one world-frame offset unless the animal faced one way. Whether
  the bars (`lattice_3d lx/ly/lz`) and the positions (`pox/poy/poz`) share
  a frame is not stated in the file or in Grieves et al. 2020's methods.
- Harness workaround: none; the lattice's verdict is a mismatch on this
  offset (its side, +0.0 % of the stated 97 cm, still matches), and the
  loader keeps the bars as stored.
- Proposed change: none in neurospatial; ask the authors whether the bars
  were placed in the tracking frame, or how
- Status: open

## G-160: A track's offset is read in a window narrower than its samples' lateral scatter

- Datasets: grosmark_000044_linear
- Operation / step: `_tracks.frame_offset` and the per-stretch bias in
  `_tracks.edge_fit` (fidelity study, declared tracks)
- Kind: harness-limitation
- Evidence: both read a median of the samples within 3 bins (7.5 cm) of
  the track. On 000044's linear track the lateral offsets of samples
  projecting inside it run from -12 to +25 cm (5th-95th percentile; median
  +7.0 cm, quartiles -0.9 and +14.7), with a dip near 0. In the window the
  median is 1.5-1.7 cm; one least-squares step of the frame fit gives
  1.35 cm, and refitting on the samples less the shift found, until it
  settles (as the fit does; needed on short corridors, where one step
  recovers only half to two thirds of a planted shift), moves the window
  with it and ends at 3.22 cm (2.95 cm with any one piece left out,
  1.18 bins), a mismatch. On every other declared track the refitted
  offset is 0.42 bin or less. The track's across-track placement comes from
  the file's linearized coordinate; no source read states where the track
  lies across it, and the reward-end samples spread up to 27 cm across the
  axis.
- Harness workaround: none; the verdict is a mismatch and the numbers are
  reported. Whether the offset should be read over a window that follows
  the fit, a fixed one, or a lateral scatter this wide should be "not
  verified" is a maintainer decision.
- Proposed change: none in neurospatial
- Status: open
