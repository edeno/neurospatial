# Appendix — dataset access details

[← back to PLAN.md](PLAN.md)

Source of everything here: the survey report
`docs/research/reports/Open spatial neuroscience datasets.md` and the notes in
`docs/research/research_notes/` (both git-ignored, local to the maintainer's
checkout). "Probed" means the survey opened that exact file. Values marked
*unverified* must be checked by the loader author before they become
expectations.

Resolve DANDI files by path at a published version:
`DandiAPIClient().get_dandiset("<id>", "<version>").get_asset_by_path("<path>")`
gives `asset.identifier` (for `dandi_asset_url`) and the SHA-256 in its raw
metadata. Pin both in the loader.

## sargolini_000582 (phase 1)

- DANDI 000582, version `0.251111.2151`, CC-BY-4.0, 1.86 GB total, files 3.7–19.1 MB.
- File: `sub-10884/sub-10884_ses-03080402_behavior+ecephys.nwb` (15.9 MB) — the pynapple null-distribution tutorial session.
- Content: rat MEC (Sargolini et al. 2006), 1 × 1 m box; position in `SpatialSeriesLED1` (and LED2), stored in cm with `conversion=0.01` (to metres), range about −50 to 50, 50 Hz; LFP 250 Hz.
- Quirk: some 000582 files hold a single unit; a session population means grouping files by subject and session. The tutorial file has six units in pynapple's table.
- Headings: two LEDs give head direction as the angle from LED2 to LED1 (or the reverse; check the dataset description) — the loader computes it, records the conversion and its convention, and checks it against movement direction during fast running (they should agree on average).
- Expectation seeds: `n_units` 6; `sampling_rate_hz` (49, 51); `position_min` ≥ (−55, −55); `position_max` ≤ (55, 55) cm; `n_dims` 2.

## peyrache_mouse32 (phase 1)

- OSF `jb2gd` (`https://osf.io/jb2gd/download`), 36,589,592 bytes, `Mouse32-140822.nwb`; license on OSF *unverified*, parent DANDI 000056 is CC-BY-4.0. SHA-256 is in NeMoS's pooch registry (`src/nemos/fetch/fetch_data.py`).
- Content (from pynapple's HD tutorial): 49 units, 31 in ADn; head direction `ry` in radians; epochs tagged `wake` and `sleep`; position x/z.
- Check the head-direction frame: neurospatial's allocentric convention is 0 = East, +π/2 = North, counter-clockwise. Determine the file's convention from the tracking description; if unknown, record that as a `data-quirk` gap and do not silently rotate.
- Expectation seeds: `n_units` 49; predicate "31 units labelled ADn"; `epoch_names` include wake and sleep; headings span about [−π, π] after wrapping.

## figure8_000552 (phase 2)

- DANDI 000552 (Huszár, Zhang, Blockus & Buzsáki), version `0.230630.2304`, CC-BY-4.0, 117 files, median 0.096 GB.
- Probed asset `61ea529d-46db-42e6-b742-17588b2f70a2`: 113 units; linearized position 0–222 cm; trials with visited and expected arm; `processing/behavior/SleepStates` and `processing/ecephys/Ripples` time-interval tables.
- Quirk: some files (e.g. the 14 MB `e13-1f1`) have only SleepStates and Ripples — no units, no position (confirmed by a full-tree walk).
- Expectation seeds: `n_units` 113; `position_min` ≥ 0 and `position_max` ≤ 222 on the linearized axis; `event_names` include ripples; `epoch_names` include sleep states.

## rosenberg_labyrinth (phase 2)

- GitHub `markusmeister/Rosenberg-2021-Repository` (MIT, repo ≈176 MB) plus a Caltech data archive (*unverified* URL — find it from the repo README). 19 mice, ≈7 h each.
- Maze: 6-level binary tree, 127 nodes, 63 T-junctions. Per-frame nose x(t), y(t) (DeepLabCut, also body/tail/feet keypoints), reduced in the paper's code to cell and node sequences.
- Format: custom "tf" files, 0.35–21 MB per mouse; the repo's Python code reads them. Read with the repo's reader logic re-implemented or vendored (MIT) — never `pickle`.
- No neurons: this dataset exercises environment-from-graph, region visits, trajectory segmentation, `graph_turn_sequence`.
- Expectation seeds: 127 graph nodes; 63 junctions; position units = whatever the repo documents (pixels or maze cells — *unverified*).

## yartsev_bat_demo (phase 2)

- Zenodo 15738988 (Forli, Fan, Qi & Yartsev 2025 demo session), CC-BY-4.0, `Demo_Session.zip` 3.3 GB; read members with `fetch_zip_member`.
- Members to fetch: `Demo_Session/Extracted_Behavior_240419.mat` (22.1 MB) and `Demo_Session/SU_kilosort4_outdir_probe{1,2,3}.mat` (24.9, 113.2, 99.6 MB). Optional: `RPL_probe*.mat` for ripples (97–420 MB). Skip LFP (≈2.2 GB).
- Behavior (MATLAB v5, opened): `r` (642,724 × 3) 3-D position in metres at `Fs` = 120 Hz; `t` seconds (−107.4 to 5,248.6); `r_lim` room bounds [[−2.9, 2.9], [−2.6, 2.6], [0, 2.3]]; `bflying` flight flag (2.4% of samples); `f_num` 38 flights; `f_smp` flight start/stop samples; `v`, `v_abs`, `v_th` 0.5.
- Spikes (opened): `out.good_units` is a MATLAB `table` → `matio.load_from_mat(path)["out"]["good_units"]` gives a DataFrame with `cluster_id`, `spikeTimes_usec` (global clock, µs; use these), `localSpikeTimes_usec` (do not use), `spikePos_um`, `template`. Probe 1: 24 good units, 37 MUA. Spike times −12.7 to 5,255.2 s, sorted per unit. Merge putative duplicates listed in `merges_Dataset_*.mat` (see `NP_Bat_Demo.m` in the record for the merge logic).
- Quirks: 0.24% NaN position samples; some y samples outside `r_lim` (1st percentile −4.23 m); negative start time.
- Expectation seeds: `n_dims` 3; `sampling_rate_hz` (119, 121); `position_min`/`position_max` within `r_lim` ± 0.2 m on x and z (y deliberately not constrained — known quirk); probe-1 good units 24.

## miniscope_002030 (phase 3)

- DANDI 002030, version `0.261016.1321`, CC-BY-4.0, 2.3 GB, files 0.005–0.11 GB. Pick the smallest file with both position and fluorescence (walk it first).
- Content: mouse CA1 miniscope, 96 cm circular arena, 2-D position in cm at 30 Hz, head direction in radians; ΔF/F only (no deconvolved events, no spikes).
- Expectation seeds: `n_dims` 2; `sampling_rate_hz` (29, 31); position extent ≤ 100 cm per axis; `activity_kind` "dff".

## giocomo_000405 (phase 3)

- DANDI 000405 (draft; Gonzalez & Giocomo), 200 assets; open-field files `sub-*/sub-*_ses-*-OF-*.nwb` 6–15 MB. Probed OF asset `627b3e66-79c1-481a-9a74-afea5c9f641c`. Draft only: pin by asset id + SHA-256, and record that the dandiset is unpublished.
- Content (full-tree walk): no `processing` modules; `acquisition/track_data` generic `TimeSeries` (65,904 × 5), description `"columns=x,y,hd,sp,ha"`, unit `"mm"`, values x −62..60, y 11..134 (cm-like), hd 0..6.28 rad; `acquisition/firing_rates` `TimeSeries` (65,904 × 46) binned rates — no spike times. The `rate` attribute holds the period (0.02).
- This dataset tests: neurospatial NWB readers on data outside `Position`/`SpatialSeries`, mislabelled units, and whether neurospatial can use binned rates instead of spike times.

## giocomo_001701 (phase 3)

- DANDI 001701 (Giocomo lab), CC-BY-4.0, 6.2 GB, files 13–104 MB. Use the smallest file that has units and position (the survey's 13 MB file; resolve by listing assets sorted by size and walking it).
- Content: MEC + CA1 (and visual areas in the probed session) Neuropixels; 320 units in the probed file; position in cm at 60 Hz; head direction labelled `"degrees"` but values in [−π, π] (radians); X-maze task trials.
- This dataset tests: `read_head_direction` trusting a wrong unit label.
