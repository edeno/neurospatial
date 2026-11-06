# Dataset loaders, round 3: awake replay

Status: draft, 2026-11-05. Builds on the loader designs of rounds 1 and 2
(`.claude/docs/plans/dataset-loaders/DESIGN.md`,
`.claude/docs/plans/dataset-loaders-round-2/DESIGN.md`), whose contracts hold.

## Why

The maintainer's words: show awake replay, and neurospatial's ability to
capture it with its methods, especially from datasets whose paper was about
replay. The datasets come from the maintainer's Zotero "Replay" collection:
the papers in it whose data are open. Frank-lab datasets are wanted but not
in this round.

## The bar (maintainer's decisions)

- A dataset loads with the automatic checks and written watch-outs, as in
  round 2.
- Candidate events are detected **the way the paper detected them**, with
  `ripple_detection.literature_methods`, which implements one detector per
  paper from its methods and code. Where that module's note marks a step
  under-specified or a code-versus-text discrepancy, the worked result and
  the watch-outs say so; nothing is tuned to reach a published number.
- Each dataset shows awake replay end to end: the paper's candidate events,
  decoded with neurospatial, judged with neurospatial's replay methods (the
  paper's own preset where one exists).
- LFP is fetched on request, as video is.

## The datasets

| Module (proposed) | Archive, license | What it holds | Candidate events |
| --- | --- | --- | --- |
| `igata_2021` | Mendeley Data 10.17632/4xk5w69yr5 (v2), CC-BY-4.0, 8.9 GB: about 1,500 per-trial `.npz` files over 15 recording days, plus `Learning_data.zip` (0.9 GB) | Rats in a 1 m lattice maze of 5 x 5 squares of 20 cm with detours and a return path (the authors' `virtual_maze_field.py`). Positions in mm and spike times in ms (the readme); 2 kHz LFP from several tetrodes; per-frame task flags (trial, delay, nose-poke, reward port, checkpoints) | `literature_methods.igata_2021` (population-rate bursts, then a Gaussian-mixture split on burst rate and ripple power; the split is marked under-specified). Preset `IGATA_2021` |
| `bhattarai_2020` | Figshare 10032866 (v2), CC0, one 659 MB zip | Rats on a maze with forced and free choices; 25 session sets: `spikes.mat` (sorted units, s), `VT.mat` (time, x, y, head direction in degrees), `events.mat` and `beh.mat` (trials, delays, rewards, choices), `lfp.mat` (one tetrode, 2 kHz, 1.1 GB) | `literature_methods.bhattarai_2020` and `bhattarai_2020_ripples`. No preset in neurospatial |
| `carey_2019` | DataLad `labs/mvdm/MotivationalT`, ODC-PDDL-1.0; 4 rats (R042, R044, R050, R064) x 6 sessions | Rats on a T-maze with food and water arms (van der Meer lab). Neuralynx/MClust: `.t` spike trains, `VT1.nvt` position (pixels; `keys.m` gives pixel-to-cm factors), `.ncs` LFP, `Events.nev`; `metadata.mat` with trial and pedestal-rest intervals, the track's centre lines in cm, 50 hand-labelled SWRs and the detector's frequency template | `literature_methods.carey_2019` (joint LFP-spectrum and multi-unit score). Preset `CAREY_2019` |
| `forli_2025` (existing) | Zenodo 15738988 | The bat demo session; the archive holds ripple tables the loader does not yet expose | The archive's ripple tables, as events. No detector |
| `huszar_2022` | DANDI 000552, CC-BY-4.0 (the survey's `figure8_000552`) | Mouse CA1 on a figure-8 maze with sleep before and after; ripples, sleep states and trials in the file; Yang et al. 2024 analysed it | `literature_methods.yang_2024`; the file's ripple table too |

Module names follow `<author>_<year>` of the paper that published the data.
DANDI 000552 is Huszár et al. 2022's dataset, which Yang et al. 2024 reused;
its worked result may come from either paper.

## Validation against the authors' own events

- **Carey:** the paper's code (github.com/vandermeerlab/papers,
  `Carey_etal_submitted`) ships the candidate events the paper used, as
  `SWRcandidates/<session>-candidates.zip`. Our `carey_2019` events on the
  loaded data are compared with them event by event. Check that
  repository's license before shipping any of its files; comparing in a
  test that downloads them needs no redistribution.
- **Yang:** the paper's analysis release (Zenodo 10.5281/zenodo.10685490)
  ships one session's events, `e13_26m1_210913.rippleHSE.events.mat`; if
  that session is in DANDI 000552, compare likewise.
- **Forli:** the ripple tables are the authors' own; nothing to compare.
- **What a comparison decides.** Agreement is measured both ways (the share
  of the authors' events our detection overlaps, and the share of ours the
  authors' overlap) and reported in the worked result. A low agreement is
  investigated until its cause is found and named (an input we do not have,
  a step the paper leaves unstated), not accepted or tuned away. The test
  pins the measured agreement, so a later change that moves it fails.

## LFP on request

`Recording` gains an LFP source, like `VideoSource`: where the LFP is, its
sampling rate, the channels and what each is (tetrode, the paper's chosen
ripple channel, for example Carey's `goodSWR` and Bhattarai's tetrode), its
unit with a `Conversion`, and `fetch()`, which downloads (or reads by byte
range) and returns the samples on the recording's clock. Nothing is fetched
at load. The summary says the LFP exists and its size.

## `ripple_detection`

- Pinned at `master` commit `0eddae64a5ebf7b810fb1ec693643b1673650142`
  (2026-10-29), in the **dev** dependency group only, until 2.0 is
  released (PyPI rejects git dependencies). The public `datasets` extra does
  not include it; code that uses it imports it optionally and says how to
  install it.
- It is used in worked results, tests and docs, never in a loader: loaders
  stay data-only.
- Its `Recording.from_arrays` takes every signal on one time grid at the LFP
  rate (multi-unit counts as an `(n_time, n_units)` array). One small adapter
  builds it from a neurospatial `Recording` and its fetched LFP; measure its
  memory on the largest session before using it everywhere.

## Settled for the builder

- **Carey's 50 hand-labelled SWRs are the detector's input,** as in the
  paper: `carey_2019` builds its spectral template from them
  (`example_ripples`). They are not a check of the detection.
- **Downloads.** `datasets.datalad.org` serves the annexed files over plain
  HTTP (fetched during the design); pin them by SHA-256 as for the other
  archives (git-annex SHA256 keys, where present, give the hash).
- **Detector inputs come from the files where they hold them,** otherwise
  from a choice the worked result states:
  - multi-unit activity is the sorted units the file holds (Igata's "all
    recorded neurons" are its sorted cells);
  - baseline and rest intervals are the loader's epochs (Bhattarai's
    pre-rest, Carey's pedestal rests);
  - place cells are chosen in the worked result with neurospatial's
    place-field detection, stated as a choice;
  - DANDI 000552's ripple table and sleep states are the `yang_2024`
    inputs; the awake candidates are those on the maze.
  - Where the file lacks an input the paper used (Bhattarai: one tetrode
    of the two), the deviation is a watch-out, and no proxy is invented.
- **Judging.** The paper's preset where one exists (`CAREY_2019`,
  `IGATA_2021`). For a paper without one, a preset is added to
  `neurospatial.replay` when its methods are read and specific enough
  (Bhattarai states 20 ms bins in 10 ms steps, line fits and 1,000 circular
  shifts, p < 0.05); otherwise the worked result says which method it uses
  and why.
- **Forli (maintainer's decision): decode in 3-D, as our own analysis.**
  The archive's ripple events are decoded in the room's voxels and judged
  there. This is not the paper's analysis and nothing claims it is: the
  paper's replay analysis was 1-D, along clustered flight paths, which
  neurospatial cannot build yet. The worked result is a sanity property of
  our own 3-D analysis, stated as such, with no comparison to the paper's
  numbers; a watch-out says that 3-D place fields here barely reproduce
  between halves of the session (0.15 in the current record).
- **Events** go in `rec.events`, which exists; no new field.
- **Clocks.** The LFP source returns samples with their own timestamps on
  the recording's clock. The adapter bins spikes and interpolates speed onto
  the LFP grid, which `ripple_detection` requires. Alignment is checked on
  data: population firing should peak at detected ripple peaks.
- **CI.** Worked results and validations are `realdata` tests (local), as in
  rounds 1 and 2; docs fences that need data are skipped with a reason.

## Goals

1. Four new loaders (`igata_2021`, `bhattarai_2020`, `carey_2019`,
   `huszar_2022`) and Forli's ripple tables as events, with registries,
   watch-outs, `EXPECTED_WATCH_OUTS` and docs pages.
2. The LFP source on `Recording`, for every dataset here that has LFP.
3. One awake-replay worked result per dataset, detecting events with the
   paper's method and judging them with neurospatial.
4. The two event-by-event validations above.

## Final audit against each paper and its code (maintainer's requirement)

Before the PR opens, each dataset's detection, decoding and replay judging is
checked step by step against the paper's text and, where public, the
authors' code, by someone who did not write that loader. For each step the
audit records what the paper says, what the code does, what ours does, and
whether they match. A difference is fixed, or stated in the worked result
with its reason. A step neither source settles is listed as unsettled.

Published code is read for comparison only where its license does not allow
adapting it. This covers `ripple_detection`'s `literature_methods` and the
`neurospatial.replay` presets as well as the loaders.

## Non-goals

- Frank-lab datasets (next).
- Boundary-vector and object-vector cells (their own round).
- Releasing `ripple_detection` (the maintainer's).
- Event detection inside neurospatial or inside loaders.
- Sleep replay as the focus; sleep epochs are loaded where the files hold
  them.

## Anti-goals

- Tuning detection or replay parameters to match a published number.
- Citing a paper, supplement or code not read; `ripple_detection`'s paper
  notes say what they read, and claims are checked against those sources.
- Unpickling outside the restricted unpickler (Igata's `spike` field is a
  pickled object array; `spike_hist`, 1 ms bins, or the restricted
  unpickler give the same spikes).
- Running `keys.m`: it is MATLAB code, parsed as text.

## Constraints

- One PR for the round.
- `ripple_detection` stays out of the public dependencies until released.
- No heavy new dependency for Neuralynx and MClust files without a reason:
  the formats are small fixed-layout binaries.
- CC-BY-SA, CC-BY, CC0 and ODC-PDDL data are downloaded by the user, not
  redistributed.

## Left to the builder

Session enumeration per archive (which Igata days have units; Bhattarai's
25 session sets; Carey's 24 sessions), the LFP source's exact API, the
adapter's shape, readers for Neuralynx and MClust files, docs layout, and
which published number (if any) each worked result reproduces.
