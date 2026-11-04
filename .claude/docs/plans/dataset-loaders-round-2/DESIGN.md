# Dataset loaders, round 2: design

Status: draft, 2026-11-04. Builds on the merged loader design
(`.claude/docs/plans/dataset-loaders/DESIGN.md`, merged as #282), whose
contracts this round keeps unchanged.

## Why

The maintainer's words: users should have access to more open datasets
that we have vetted; the datasets should show off the package's features;
and they should prove the package works on real recordings. Warning users
when data looks unusual is a core feature the datasets should also show.

The six datasets in `neurospatial.datasets` (Comrie 2026, Grosmark 2016,
Peyrache 2015, Sargolini 2006, Forli 2025, Valero 2022) leave gaps against
what the package does:

- **Grid cells with a number to compare.** Sargolini has grid cells but no
  headings and no worked result (the paper could not be read).
- **Head direction with position.** Peyrache has head direction but no x/y,
  so heading cannot be checked against movement and conjunctive coding
  cannot be shown.
- **The animal's frame and objects.** Nothing has objects.
- **Graph mazes beyond J16's bandit maze.** No radial maze or cheeseboard.
- **3-D structure.** The bat flies in an open room; nothing has a climbing
  structure.

## The bar for a new dataset (maintainer's decision)

A dataset is added when it **loads with the automatic checks and written
watch-outs**. A reproduced published number is welcome but not required;
`WORKED_RESULT` may be `"not possible yet"` with its reason, as for
Sargolini, Forli and Valero.

Everything else in the round-1 design holds: every session of an archive
is offered and the sessions examined by hand are marked verified; values
carry a basis (`stated` or `inferred`) with evidence; clock and unit
failures refuse unless `accept_failed_checks` names them, other checks
warn; data are never altered by a chosen rule, only described (the
warn-don't-alter rule); only sources that were read are cited; downloaded
files are untrusted (no pickles except through the survey's restricted
unpicklers, pinned checksums).

## The datasets

All eight already open in the survey harness (`datasets/load_*.py` in the
repo root), whose docstrings hold the evidence for every conversion and
watch-out below. The public loader ports that evidence; it does not
re-derive it.

| Module | Archive, license, access | What it shows off | Quirks the checks should report |
| --- | --- | --- | --- |
| `vollan_2025` (open field) | EBRAINS 10.25493/R5FR-EDG, CC-BY-SA-4.0, public bucket, no login; `sharing_v4/navigation/of/26018_2.mat`, 205.7 MB, MATLAB v7 via `mat-io` | Grid modules with the authors' per-module grid scores (median 1.11 / 0.99 / 0.68); head direction with position, frame verified against movement | Authors removed every sample under 5 cm/s, kept as gaps: the masked-time warning fires. Circular 150 cm arena inferred from the session type and the samples' shape, since the paper names no file |
| `vollan_2025` (linear track) | Same archive; `sharing_v4/navigation/lt/27764_1.mat`, 110.6 MB | MEC units on a 200 cm track with OptiTrack head direction; one weak grid module (median 0.26) | Same speed-filter gaps; the file's session table ends at 7486 s but samples run to 8412 s |
| `gardner_2022` | Figshare 16764508, CC0; two members of a 320 MB zip by byte range (169 MB `.npz` + sessions text) | 483 units in grid modules 1–3, 1.5 m box, head azimuth with position | Spike trains are pickled object arrays: read only through the restricted unpickler that lists each pickle's imports first. Only the open-field session's valid ranges are analysis epochs |
| `peyrache_rsc` (DANDI 000987) | DANDI 000987, CC-BY-4.0; smallest file with units and position 20.2 MB | Retrosplenial units, sleep then a square arena, motion-capture head direction with position | Blank units and reference frame (axis roles inferred from ranges; metres inferred); held-frame runs and fast steps reported by `find_suspect_tracking`; head-direction frame **not verified** by movement (+0.5 rad offset, 49° spread) |
| `kanter_2025` | EBRAINS 10.25493/GTVB-FQR, CC-BY-SA-4.0, public bucket, no login; novel-object session 29.6 MB | LEC and CA1 Neuropixels units, empty arena then three novel objects placed one at a time (object epochs) | Unit ids repeated across two probes (380 rows, 235 ids): `read_units` refuses, the loader names units `<area>-<id>`. Stated reference frame contradicted by the values. Head-direction frame not verified and its offset drifts. **Object positions are not in the files**, so object-vector analysis cannot run |
| `shipley_2026` | UCL RDR 30902972, CC-BY-4.0; four members of a 4.4 GB zip by byte range (~67 MB) | Mouse CA1 on an 8-arm radial maze (declared star graph from the stated arm sizes), 11 trials, sleep | Position times rebuilt per trial from the spike-to-sample map (camera ran at 30.05 Hz, not the header's 30 Hz; residual ≤ 41 ms), an inferred conversion; `pos.dir` is displacement, not head direction |
| `huszar_2026` (DANDI 001880) | DANDI 001880, CC0-1.0; smallest file with units and position 65.7 MB | CA1/CA3 units on a cheeseboard maze, PRE / learning / POST, 45 learning trials, reward coordinates, 2,118 ripples | 29 % of frames held in long runs (`find_suspect_tracking`); environment inferred from samples (no stated size read); two position series, the loader names one |
| `grieves_2020` | UCL RDR 10.5522/04/9926477, CC0; one Deflate64 member (431 MB) of an 11.2 GB zip by byte range | CA1 place cells in a 3-D diagonal climbing lattice, with the lattice's 147 bars as geometry | Position unit undocumented: mm inferred from the stored velocity (ratio 10.03); headings are the overhead camera's azimuth in a mirrored frame; needs `inflate64` |

Module names are proposals; the round-1 pattern is `<first author>_<year>`,
and two datasets here are named by lab and dandiset because the file's
paper is not the dataset's (000987) or could not be read (001880).

### Showing the warnings

Each loader's `WATCH_OUTS` already describes what a reader must know. This
round adds, per verified session, the list of load warnings it is expected
to raise (for example Vollan: masked time; 000987: suspect tracking and an
unverified heading frame), and a test that loads the session and checks
each still fires. Real files become regression tests for the checks: a
check that stops firing on a file known to have the problem is a bug.

## Goals

1. The eight loaders above in `neurospatial.datasets`, through `load` and
   `catalog`, with registries, watch-outs, expected warnings, and the
   overview figure and animation working on each.
2. Every session of each archive offered; the survey's session per dataset
   is the default and is verified by hand.
3. The catalog page and CHANGELOG list the new datasets with what each
   shows off.

## Non-goals (next rounds)

- **Replay.** Next round, from papers about replay, preferring awake
  replay: Tirole/Bendor 2022 (Zenodo 7114733), Ólafsdóttir/Barry 2016
  (Zenodo 5566548), Shipley's replay analysis, Bhattarai/Jung 2020 (Figshare
  10032866, unvetted).
- **Boundary-vector and egocentric boundary cells.** Next round; no public
  recording from the original papers was found, and the candidate
  substitutes (van Wijngaarden/Ito, Zhang 2022) are unvetted.
- **Frank-lab datasets** (000447 W-tracks, hc-28, Denovellis 2021): wanted,
  not this round.
- **Phase precession.** Deferred; no candidate's LFP was checked.
- **New analysis features.** Gaps a dataset exposes (for example Vollan's
  module-3 grid score) are recorded as watch-outs or follow-ups, not fixed
  in this round.

## Anti-goals

- Masking, dropping or correcting samples by a chosen rule.
- Citing a paper or readme that was not read.
- Unpickling outside the restricted unpicklers; `allow_pickle=True`.
- Any login: both EBRAINS buckets are public, and no dataset here needs
  CRCNS.

## Constraints

- One PR for the round.
- Public API changes go through `tests/data/public_api.txt` and
  `.claude/rules/public-api.md`; the loaders add modules, not new types.
- New dependencies (`mat-io`, `inflate64`) go in the `datasets` extra.
- CC-BY-SA data are downloaded by the user at load time, never
  redistributed; the summary states the license as the archive does.

### Kanter's objects (maintainer's decision)

Kanter ships for its LEC and CA1 units, object epochs and warnings. The
builder also looks for the objects' locations in the archive's other files
and any video. A location that is found becomes `object_positions` with
its source as evidence. A location read from video is `inferred` and needs
a stated pixel-to-cm calibration. If none is found, `object_positions`
stays unset, the watch-out says why, and the object-vector goal moves to
the boundary-cell round.

## Settled for the builder

From the round-1 design and code:

- **Readers exist.** `_fetch.fetch` downloads any public HTTPS file
  (both EBRAINS buckets are plain HTTPS); `fetch_zip_member` reads a zip
  member by byte range, Deflate64 included; `open_nwb` streams DANDI.
  `mat-io` and `inflate64` are already in the `datasets` extra.
- **One module per dataset, one session per file.** Vollan's open-field
  and linear-track files are sessions of one `vollan_2025` dataset.
- **The public loaders replace the survey's.** As in round 1, each survey
  `load_*.py` for these eight imports the public loader, so there is one
  loader per dataset.
- **Real-data tests use the existing `realdata` marker** (local, not in
  default CI). Expected-warning tests are `realdata` tests; the synthetic
  tests check each loader's conversions on small fabricated files.
- **Unverified heading frames keep the stored values**, wrapped to
  [-π, π], with an `inferred` `Conversion` and a watch-out, as Peyrache
  2015 does. The overview and summary already say when a frame is not
  verified by movement.
- **Grieves' frame change is a stated conversion,** not a chosen rule:
  the survey derived it from the file (the 3-D frame is the camera's
  mirrored in y and shifted), so the loader applies it with that evidence.
  The 3-D environment is a voxel grid as for Forli; the lattice bars are
  kept as geometry for plots.

Calls made in this document (the maintainer may overrule):

- **Verified** means the developer examined the session by hand: the
  survey's evidence re-checked on the public loader's output, plus its
  overview figure and animation looked at. The default session per dataset
  is examined; other sessions load with the automatic checks only. The
  maintainer confirms in the PR review.
- **Checksums:** pinned for every offered session. DANDI publishes
  SHA-256; for Figshare, Zenodo, UCL RDR and EBRAINS the files are
  downloaded once during development to pin them (the archives total well
  under the existing 65 GB cache).
- **All eight ship in the one PR;** none is dropped to save time.
- **Clock and range findings** (Vollan's session table ending before the
  samples) go through the existing checks unchanged; whatever they report
  on the default session is recorded as a watch-out, and the PR states
  whether any default session needs `accept_failed_checks`.
- **Names:** `peyrache_rsc` for DANDI 000987 (the file's paper is not the
  dataset's); every other module follows round 1's `<author>_<year>`, with
  the paper's year (`shipley_2026`, `huszar_2026`).

## Left to the builder

Session enumeration per archive, test fixtures, the per-dataset `params`,
how expected warnings are declared, docs layout.
