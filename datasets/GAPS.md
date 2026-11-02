# Gap log

One entry per finding, numbered in order of discovery. `Kind` is one of
wrong-result, missing-feature, friction, error-message, docs, performance,
data-quirk, false-alarm, silent-hazard. Results referred to below are in
`results/` (`matrix.md`, `warnings.md`, `reference_checks.md`, `<key>.json`).

## G-001: Spike-drop warning fires when spikes fall outside tracking but also outside the requested epochs

- Datasets: peyrache_mouse32
- Operation / step: hd.tuning (`compute_directional_rates(..., epochs=wake)`)
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
  keep that.
- Harness workaround: none (the warning is recorded, the op is `ok`)
- Proposed change: count as dropped only spikes inside the requested
  `epochs ∩ spike_window` that fall outside tracking; keep the clock-ratio hint
  for the units case; add `warn_on_drop=` to the directional entry points.
- Status: open

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
- Proposed change: undecided — make `binned` mean unsmoothed by default, or
  keep the default and say in the `method` docs and the result's repr that
  "binned" is smoothed.
- Status: open

## G-003: Spikes are assigned to the tracking sample at or before them; pynapple uses the nearest sample

- Datasets: sargolini_000582 (reference check 1)
- Operation / step: reference check, spatial information vs pynapple
- Kind: docs
- Evidence: per-bin spike counts: neurospatial matches the at-or-before rule
  exactly (0 of 4013 spikes in a different bin) and pynapple the nearest-sample
  rule (1 of 4013); the two rules place 218 spikes in different bins at 50 Hz
  tracking, which moves spatial information by up to 2.1 %.
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
  wake) gives a median absolute error of 9.5° against the tracked heading,
  with the tutorial's 19590 decoded samples reproduced.
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
  the values differ by up to 0.178. A threshold near 0.3–0.4 would classify
  unit 2 differently.
- Harness workaround: none
- Proposed change: undecided — compare the overlap rule and NaN handling with
  BNT/opexebo's `normxcorr2_general`, then either match it or document why not.
- Status: open

## G-007: Position decoding with six MEC units is at chance; silent bins all decode to one corner

- Datasets: sargolini_000582
- Operation / step: decode.position
- Kind: docs
- Evidence: median error 66.0 cm with 250 ms bins in a 100 cm box, where
  shuffled-pair chance is 50.6 cm. 40 % of bins have no spikes; every one of
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
  `sub-10884_ses-03080402` (full tree walked); the survey appendix expected two
  LEDs. `hd.tuning` is `n/a` for this dataset.
- Harness workaround: none
- Proposed change: none; a later batch can use a two-LED 000582 session.
- Status: open
