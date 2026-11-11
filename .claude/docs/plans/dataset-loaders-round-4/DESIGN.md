# Dataset loaders, round 4: replay reproductions for the non-local paper

Status: building, 2026-11-11. One PR (branch `feat/replay-reproductions`).
Builds on rounds 1-3 (`.claude/docs/plans/dataset-loaders*/DESIGN.md`); their
contracts hold: plain arrays, every session offered, checks that refuse and
watch-outs that warn, inferred facts labelled, one worked result per dataset,
`realdata` tests, LFP on request, `ripple_detection` used only outside loaders.

## Why (maintainer, 2026-11-11)

Datasets people can access easily, that test and demonstrate the metrics, and
that reproduce published results where possible. For the non-local paper:
the published replay-detection methods in place (detection in
`ripple_detection.literature_methods`, judging in `neurospatial.replay`
presets), reproducing all or some of each paper's results on its own data.
Outside the Frank lab; awake replay.

## The bar for each dataset

1. **Loader** in `neurospatial.datasets` with registry JSON, watch-outs,
   `EXPECTED_WATCH_OUTS`, docs page (`docs/datasets/<id>.md`, mkdocs nav),
   catalog entry, synthetic tests and `realdata` tests.
2. **The paper's method, checked against the authors' own output** where we
   have it (below). Agreement is measured both ways and pinned in a
   `realdata` test; a gap is investigated until its cause is named, never
   tuned away. Where the authors' code and text differ, the code wins for
   intentional choices; clear bugs are documented, not reproduced in the
   package (they may be reproduced in a test helper to show the match).
3. **A reproduced number** from the paper, pinned, with the criterion fixed
   before the run (WORKED_RESULT).

## Datasets and references

| id | Archive | Reference (authors' own output) |
| --- | --- | --- |
| `carey_2019` (exists) | DataLad MotivationalT | Authors' MATLAB run in Octave: 3,657 sequences over the 19 paper sessions (paper 3,780); `.claude/docs/carey-octave-repro/` |
| `tirole_2022` (new) | Zenodo 7114733 (CC0; mirror of Dryad 10.5061/dryad.ksn02v76h, whose API needs a token) | Authors' MATLAB run in Octave: per-event scores/p-values and significant RUN replay; Fig. 3A RUN regression B = 0.0502, R^2 = 0.430, n = 227 (paper B = 0.05, R^2 = 0.41, F(1,268) = 186); `.claude/docs/tirole-octave-repro/` |
| `pfeiffer_2013` (new) | kemerelab/UncoveringTemporalStructureHippocampus `data/openfield.nel` (MIT), one session: Pfeiffer & Foster 2013 rat 1 day 1 | Pfeiffer's 322 `RippleTimes` and 110 `SignificantRipples` in the file; Krause & Drugowitsch 2022 metadata confirms rat1 day1 n_SWRs = 322; `.claude/docs/pfeiffer-openfield/` |
| `shin_2019` (new) | DANDI 000978 (CC-BY-4.0), 9 NWB files, 28-47 GB each, mostly raw | Text: per reward-well visit 10.3 s immobility, 7.3 SWRs, 2.3 candidates, 0.9 replays; 818/1,533 trials with candidates |
| `olafsdottir_2016` (new) | Zenodo 5566548 (CC-BY-4.0), Axona, 9.9 GB zip | To establish from the paper (Zotero PDF) |

## Settled facts (verified 2026-11-10/11, do not re-derive)

- **Tirole deposit:** the clusters/position and CSC labels are crossed.
  Pairing by clock span, confirmed by Table 1 lap counts (`extract_laps`
  counts one-way runs, so divide by 2):
  - clusters+position `RAT1_SESS1` + CSC `RAT4_SESS3` = Rat 4 session 3;
  - clusters+position `RAT1_SESS2` = `RAT4_SESS1` (byte-identical) + CSC
    `RAT1_SESS1` = Rat 1 session 1;
  - CSC `RAT4_SESS1` = Rat 4 session 1, no spikes in the deposit;
  - the other 7 sessions are labelled correctly. Table in
    `.claude/docs/tirole-octave-repro/results/PAIRING.md`.

  Every CSC except RAT2_SESS1 is MATLAB v7.3 (HDF5 struct array of
  references; empty fields carry `MATLAB_empty`). The deposit has no
  multi-unit clusters (the paper's candidate MUA used them), no waveforms (no
  pyramidal half-width filter), no `best_CSC.mat` (use the CSC channel
  labels). RAT2_SESS1's CSC ends at 100,629.6 s, before its re-exposure
  tracks 3-4.
- **Pfeiffer file:** a gzip dill pickle of nelpy 0.1.2 objects. Read without
  executing; numpy arrays come through `dill._get_attr(<numpy module>,
  '_reconstruct')`. 212 units (203 excitatory, 9 inhibitory), 1,201,998
  spikes, 2572-4759 s; 2-D position at 57.4 Hz, x 91-290, y 1-193 (cm,
  inferred: a 2 m square); 100 Hz MUA; 322 `BradRippleStartEndTimes`; 110
  `BradSignificantRippleIDs`, 0-based into those 322 (offset 0 alone matches
  Maboudi's HMM significance, coherent p = 4e-5). Krause loads MATLAB
  `SignificantRipples - 1`. The labels' criteria are not documented in the
  file; P&F 2013 reports 274 trajectory events for R1 D1 (89 home + 185
  away), so the 110 are not that set.
- **Shin files:** `units` (sorted, spike_times, electrodes), position
  `processing/behavior/Position`, LFP `processing/ecephys/LFP`,
  `intervals/epoch` (17), `intervals/trials` (start_well, end_well,
  trajectory_type, correct); `acquisition/ElectricalSeries` is raw (~14 GB),
  never read. The dandiset names no paper; the link is inferred from the 6
  rat IDs (ER1, KL8, JS14, JS15, JS17, JS21) and the task.
- **Ólafsdóttir zip:** per recording (`track1`, `sleepPOST`, `screening`):
  Axona tetrode files `.1`-`.16` (header then 216-byte records:
  `t,ch1,t,ch2,t,ch3,t,ch4`, 4-byte big-endian timestamps at the header's
  `timebase`, 50 int8 samples per channel), `.clu.N`, `_N.cut`, `.pos`
  (50 Hz), `.eeg` (250 Hz), `.egf` (4.8 kHz), `.set`. Every detected spike
  has its waveform, so clusterless marks are available. Some file labels are
  inconsistent (`20141213_R2117_screening` in the r2217 folder, "Screening
  R2168"): pair recordings by content, not names.

## Rules for builders

- Read the paper (local Zotero first, `~/Zotero/zotero.sqlite`, copy it and
  open read-only) and cite only what was read.
- Follow the paper's code over its text for intentional choices; never
  contact authors.
- Warn, do not alter data: quality checks describe; nothing is NaN'd or
  dropped by a chosen rule.
- Data are downloaded by the user, not redistributed. Small reference
  tables derived from the authors' code outputs (event times, scores,
  p-values) may be committed under `tests/data/` with a note on how they
  were made.
- Tests must check what they claim: realdata tests pin measured numbers;
  synthetic tests build tiny files in the archive's format.
- Run pytest with `-n 4` at most and `MPLBACKEND=Agg`; full suite only at
  the end. In a worktree, confirm `neurospatial.__file__` points into it.

## Non-goals

Running Krause's dynamics models; sleep-replay results; Frank-lab data.
