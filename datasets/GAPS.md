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
  the W-track: extent 92.5 → 145 cm, 158 s of occupancy at the origin,
  running segments 487 → 455, all silent. Check 1 would fire (15.5 % at the
  origin against 0 % on every clean recording, now 19 of 19 with positions).

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
  information is 3.38 bits/spike; with each unit's spikes circularly shifted
  (4 seeded shifts, `rates.information_vs_shifts`) it is 3.55, and only 34 %
  of units beat every shift (all 6 on 000582, 94 % on the figure-8).
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

## G-043: On Vollan's speed-filtered data, real units carry less spatial information than their circular shifts

- Datasets: vollan_lt, vollan_of
- Operation / step: rates.information_vs_shifts
- Kind: wrong-result (candidate; cause not found)
- Evidence: open field (rat 26018, 644 units, 233 in grid modules): median
  information 0.436 bits/spike, median over 4 shifts 0.596; 22 % of units
  beat every shift. Linear track: 0.289 vs 0.424, 12 %. On every other
  recording real units are far above their shifts (000044: 0.83 vs 0.08).
  The grid scores of the same units match the authors' module medians on
  modules 1–2, so spikes and positions are aligned. Not the cause: bridging
  of the speed-filter gaps (with `max_gap=0.025` s, 2.5 sampling intervals,
  0.431 vs 0.589, 24 %). Untested: whether the shift rule, which shifts over
  the whole span including the removed stretches, moves spikes into a few
  tracked samples; and whether firing locked to slow periods matters.
- Harness workaround: none
- Proposed change: undecided; find the cause before trusting shuffle verdicts
  on speed-filtered data
- Status: open

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

## G-092: hc-28's LED angle and the authors' head direction do not follow the path

- Datasets: hc28_hpa
- Operation / step: loader (heading frame check)
- Kind: data-quirk
- Evidence: on HPa day 1 epoch 4, the angle from one LED to the other minus
  the direction of movement above 10 cm/s has circular mean −25.5° and median
  spread 51.0° (the other order: 154.5°, 51.0°). The authors' own `pos`
  `dir` column against their smoothed x, y: −21.9°, 28.0°. The offset
  depends on the direction of movement (−85° to +22° across 45° sectors), so
  it is neither a frame rotation nor a fixed LED mounting angle that a
  constant could correct. The LEDs are 15.7 cm apart (median); `pos` x, y
  follow one LED (`arg` `diode` 1), 7.9 cm from the midpoint.
- Harness workaround: the loader keeps positions and drops head direction
- Proposed change: none for neurospatial
- Status: won't fix (data quirk)

## G-093: hc-31 stores position in inches and time in microseconds; its interneuron marker cannot exist

- Datasets: hc31_anm190422
- Operation / step: loader
- Kind: data-quirk
- Evidence: `posMazeLin`/`posMaze` give `unitsPerCm` 0.3937 and `px` 2.54:
  the coordinates are inches (the 48 m track spans 0–1896 units, 4815 cm
  after conversion); times and spike `timeStamps` are microseconds
  (`unitsPerSecond` 1e6); the velocity column is inches per microsecond
  (at most 5.8e-5). Nothing but those two scalars says so. The readme says
  interneurons and poor units have NaN in `ID` and gives the rule
  `all(ID <= 2, 2)` to exclude them, but `ID` is uint8 (values 0, 1, 2), which
  has no NaN: all 63 units pass the rule, so interneurons cannot be told
  apart from silent cells.
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
- Harness workaround: the loader sets those frames' heading to NaN
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
  warning on `rates.spatial` whenever the probe rebuilds the environment
  (three `warned` rows whose real response was silent or a different
  warning).
- Harness workaround: none (the harness's position unit is "pixel")
- Proposed change: one unit table; the registry check accepts every alias
  that `_units` resolves
- Status: open

## G-096: A missing-sample marker other than NaN or 0 passes silently

- Datasets: hc3_ec013_669, hc2_ec013_527, hc18_train261 (hazards `whl_minus_one_kept`, `pos_minus_one_kept`)
- Operation / step: env.from_samples, env.occupancy, rates.spatial, behavior.segment_by_velocity, behavior.laps
- Kind: silent-hazard
- Evidence: Neuroscope `.whl` and `.pos` files mark untracked frames with
  `-1` (hc-2 description). Kept as a coordinate, every untracked frame sits
  at (-1, -1) px, outside the video image: hc-3 540 → 553 bins and 8 s of
  occupancy there, median peak 2.99 → 3.47 Hz; hc-18 69 s at (-1, -1) and
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
