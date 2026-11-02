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
- Status: promoted (UX fixes phase 6, branch `fix/ux-error-gaps`; the ms-clock case must still warn; at most one discarded-data warning per call)

## G-002: `method="binned"` smooths with a 5 cm bandwidth unless `bandwidth=0` is passed

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
- Harness workaround: `datasets/reference_checks.py:_binned_rates` passes `bandwidth=0.0`
- Proposed change: `method="binned"` defaults to `bandwidth=0` (no smoothing)
  everywhere binned is accepted (maintainer decision).
- Status: promoted (UX fixes phase 3b, branch `fix/ux-defaults-and-returns`)

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

- Datasets: sargolini_000582
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
- Status: open (linked to G-001)

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

- Datasets: figure8_000552
- Operation / step: loader (`read_position`)
- Kind: error-message
- Evidence: with two position series, `read_position(nwbfile)` refuses to
  choose (good). `position_name="LinearizedPosition"` (the container's name)
  then raises "Position series 'LinearizedPosition' not found … pass
  position_name= with a name or path from this list", and the list holds only
  full series paths. The container name is what a user sees first in the file.
- Harness workaround: the loader passes the full series path
- Proposed change: accept a `Position` container name when it holds one series.
- Status: open

## G-013: `env.occupancy(min_speed=)` needs `speed=`; the rate and decoder entry points compute it

- Datasets: figure8_000552, rosenberg_labyrinth
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
- Status: promoted (UX fixes phase 3b: every sibling taking min_speed derives speed the same way)

## G-014: A length unit off by 100 on a 1-D track builds 3 bins without a warning

- Datasets: figure8_000552 (hazard probe `length_unit_off_by_100`)
- Operation / step: env.from_samples, rates.spatial, behavior.segment_by_velocity
- Kind: silent-hazard
- Evidence: positions ×0.01 (metres labelled cm) on the 222 cm track give a
  2.2 "cm" track: `Environment.from_samples(bin_size=2)` builds 3 bins with no
  warning, rate maps flatten (median peak 4.67 → 1.36 Hz) and velocity
  segmentation finds 0 running segments instead of 395, all silently. On
  000582 the same hazard warns because the bin is at least the whole extent
  in every dimension; here the extent (2.2) is just above the bin (2.0).
- Harness workaround: none
- Proposed change: warn when the environment has fewer than a handful of bins
  along a dimension (the existing check only fires at one bin or fewer).
- Status: open

## G-015: `detect_laps` finds laps on a maze with no loop

- Datasets: rosenberg_labyrinth, sargolini_000582
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
  one clockwise "lap".
- Harness workaround: none
- Proposed change: test for a loop on the track graph (or ignore cycles that
  stay within one junction's bins) before counting laps.
- Status: promoted (UX fixes phase 6 fix round)

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

- Datasets: rosenberg_labyrinth, yartsev_bat_demo (hazard probe `missing_positions_as_zero`)
- Operation / step: behavior.segment_by_velocity, behavior.turn_sequence
- Kind: silent-hazard
- Evidence: 4.9 % of the labyrinth's frames have no nose keypoint (NaN). With
  them set to (0, 0), a corner of the maze, velocity segmentation finds 2936
  movement segments instead of 2637 and the turn sequence 4604 turns instead
  of 4415, with no warning. On 000582 the same hazard touches one sample and
  is undetectable; here the zeros are 12,400 samples at exactly (0, 0). On
  the bat (0.24 % missing), the zeros land at (0, 0, 0), below the lowest
  tracked height (0.28 m): the environment loses 3 bins and the median mean
  rate in flight drops 4 % (0.945 → 0.909 Hz), with no warning.
- Harness workaround: none
- Proposed change: warn when a sizeable share of samples sit at exactly the
  same coordinate (or at exactly 0) between moving samples.
- Status: open

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

## G-021: Laps cannot be counted on a linearized figure-8

- Datasets: figure8_000552
- Operation / step: behavior.laps
- Kind: friction
- Evidence: the file gives the figure-8 only as a linearized 1-D position
  (and camera x, y), no track graph. `detect_laps` on the 1-D environment
  raises "this environment has no loop … for back-and-forth runs … use
  detect_trials", which is right for the input. Counting laps around the
  figure-8 needs a graph of the maze, which the dataset does not provide.
- Harness workaround: none (cell is `err`)
- Proposed change: none
- Status: won't fix (behaves as documented)

## G-022: In sparse 3-D occupancy, spatial information sits at the shuffle level while its bias estimate is small

- Datasets: yartsev_bat_demo
- Operation / step: rates.spatial_information, cells.place_screen
- Kind: wrong-result (candidate; needs a proper shuffle test)
- Evidence: 3-D rate maps of 289 units over 1352 bins of 0.2 m, restricted to
  the 38 flights (129.6 s of occupancy; median 0.067 s per visited bin, 63 %
  of visited bins under 0.1 s). Median spatial information is 3.45 bits/spike
  and median peak rate 71 Hz. One random circular shift of each unit's spikes
  gives a median of 3.61 bits/spike, so the median unit carries no more
  information than chance by this measure. `spatial_information_bias`, the
  "expected untuned information" that `classify()` compares with, has a
  median of 0.37 bits/spike; it appears to miss most of the sampling bias of
  such sparse occupancy. `min_occupancy` defaults to 0 s, so every visited
  bin counts.
- Harness workaround: none
- Proposed change: undecided — test `spatial_information_bias` against a
  shuffle null on sparse occupancy; warn when most visited bins hold very
  little time, and point to `min_occupancy` or larger bins.
- Status: open

## G-023: 3-D position decoding is near chance on the bat session

- Datasets: yartsev_bat_demo
- Operation / step: decode.position
- Kind: docs
- Evidence: training on even flights and testing on odd ones (250 ms bins,
  0.2 m bins), the median error is 2.19 m against a chance level of 2.68 m.
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
