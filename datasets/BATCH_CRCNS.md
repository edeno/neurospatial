# CRCNS batch: six hippocampal recordings

Six CRCNS recordings loaded through the harness, with what each exercised,
the cross-check of hc-11 against DANDI 000044, new gaps, warning-calibration
numbers, and the changes this batch suggests for [FINDINGS.md](FINDINGS.md)
(not edited here). Evidence is in [GAPS.md](GAPS.md) (G-090 to G-098, and
added evidence on G-014, G-015, G-017, G-022, G-037) and in
[results/](results/).

## Access

CRCNS serves files from the NERSC portal after a login form; the loaders post
it with the account from `CRCNS_USERNAME` and `CRCNS_PASSWORD`
(`neurospatial.datasets._fetch.fetch_crcns`). No bot check or browser step appeared. Most sets are
`.tar.gz` archives, which cannot be read by byte range, so
`neurospatial.datasets._fetch.fetch_crcns_tar_members` streams an archive once and keeps only the
pinned members (each by SHA-256). Throughput was about 30 MB/s for one
download and 2–13 MB/s each with three in parallel.

The SHA-256 pins came from a first download made with
`studies/crcns_first_download.py`, which streams the whole file once,
compares its MD5 with the set's published `checksums.md5`, and moves the
wanted members into the cache only on a match. All six downloaded files (five
archives and `resSave.mat`) matched:

| CRCNS file | MD5 in `checksums.md5` | Bytes streamed | Match | Members (regex argument) |
| --- | --- | --- | --- | --- |
| `hc-11/data/NoveltySessInfoMatFiles.tar.gz` | `2666ee13c8c1e3609c08c6f4ba444d11` | 277.5 MB | yes | `Achilles_10252013_sessInfo\.mat$` |
| `hc-3/ec013.38/ec013.669.tar.gz` | `dc3a6538ce24219e4b35b83faead75c9` | 233.8 MB | yes | `ec013\.669\.(clu\.\d+\|res\.\d+\|whl\|xml)$` |
| `hc-2/ec013.527/ec013.527.tar.gz` | `7db89776bb62759f8c5b2738d71e5cb2` | 211.8 MB | yes | `ec013\.527\.(clu\.\d+\|res\.\d+\|whl\|xml)$` |
| `hc-28/data/HPa_data.tar.gz` | `ca6a313b9936db3aff62f60597da34a3` | 2822.1 MB | yes | `^HPa_data/HPa[a-z]+\d*\.mat$` |
| `hc-31/data/resSave.mat` | `f97e58107878c688854474519fa6be72` | 166.8 MB | yes | (the file) |
| `hc-18/data/Train-261-20140617.tar.gz` | `31f4ca8bc7b0bb1b81ea3bf63e2d8971` | 6456.9 MB | yes | `\.(clu\.\d+\|res\.\d+\|pos\|cat\.evt\|xml)$` |

Each row was one run of
`uv run --group datasets python datasets/studies/crcns_first_download.py <file> <md5> <cache subdir> '<regex>'`
(the script was committed after the run, with the cache path taken from
`neurospatial.datasets._fetch.cache_root` instead of written out). The hc-3 cell table,
`hc-3/docs/crcns-hc3-metadata-tables.zip`, was downloaded on its own and its
MD5 (`80c99fe9c82e857be409022c97edaa9f`, `md5 -q`) matched `checksums.md5`
before its SHA-256 was pinned.

| Key | Set and session | What was downloaded | Kept |
| --- | --- | --- | --- |
| `hc11_achilles` | hc-11, Achilles_10252013 | `NoveltySessInfoMatFiles.tar.gz`, 277.5 MB | `Achilles_10252013_sessInfo.mat`, 38.5 MB |
| `hc3_ec013_669` | hc-3, ec013.38/ec013.669, linear track | session archive, 233.8 MB; metadata tables, 1.1 MB | `.res/.clu` ×8, `.whl`, `.xml` (5.5 MB), cell table |
| `hc2_ec013_527` | hc-2, ec013.527, 180 cm open field | session archive, 211.8 MB | `.res/.clu` ×4, `.whl`, `.xml` (6.3 MB) |
| `hc28_hpa` | hc-28, HPa day 1 epoch 4, W-track | `HPa_data.tar.gz`, 2.8 GB streamed whole (files sit after the LFP) | day-1 `rawpos`, `pos`, `spikes`, `task`, `tetinfo` (21.3 MB) |
| `hc31_anm190422` | hc-31, resSave(5) run 2, 48 m track | `resSave.mat`, 166.8 MB | the file |
| `hc18_train261` | hc-18, Train-261-20140617, model train | session archive, 6.5 GB streamed whole (516 s) | `.res/.clu` ×8, `.pos`, `.cat.evt`, `.xml`, 134.8 MB |

Not loaded:

- **hc-13** (rawpos zeros): one 8.1 GB `data.tar.gz` and no per-session
  file. Left out of reach as planned. Streaming it would take about 11 min
  at the 12.5 MB/s hc-18's 6.5 GB archive achieved (516 s), or about 5 min at
  the 30 MB/s of a single download, so it is reachable if wanted; its
  defect (0 for untracked frames) was tested on hc-28's `rawpos` instead, which
  has the same documented convention.
- **hc-6**: hc-28 covered the W-track (`pos` + `cmperpix`) case.
- A second hc-3 session (open field ec013.459) was not loaded: hc-2's
  ec013.527 is an hc-3 session too (topdir ec013.32), so it covers the open
  field.

## What each recording exercised

- **hc11_achilles.** MATLAB v7.3 through h5py (arrays transposed back),
  positions in metres, five scored sleep states, population-burst replay
  candidates by the 000044 rule. All 8 expectations passed. The cross-check
  below.
- **hc3_ec013_669.** Neuroscope `.res/.clu/.whl/.xml`, units restricted to
  hc-3's cell table, two-LED head direction on a linear track, pixels with no
  stated scale (0.87 cm/px inferred from the 250 cm track, used only for bin
  size and speed threshold). The track is diagonal in the image. The LED order
  is not documented; the movement check picks it (mean 6.2°, spread 17.8°;
  the other order −173.8°). 5.1 % of frames have both LEDs at one point and
  no direction (G-094). Expectation `n_units` failed: 72 of the cell table's
  125 units fire in this session (G-091).
- **hc2_ec013_527.** The same code on an open field: every cluster ≥ 2 kept
  (37), 0.995 cm/px inferred from the 180 cm side, LED order by movement (mean
  0.4°, spread 42.5°). All expectations passed.
- **hc28_hpa.** Frank-lab nested MATLAB cells (day > epoch > tetrode > unit),
  `rawpos` zeros (15.5 % of frames) as NaN, `cmperpixel` (named `cmperpix`
  in the documentation). Head direction dropped: the LED angle sits about
  −15° to −20° off the direction of movement with a residual of about 30°
  (−14.4°, spread 18.8°, over 15-frame steps), outside the check's 0.2 rad
  at every lag (G-092); with one-frame steps the whole-pixel positions
  inflated the spread to 51° (G-098, `heading_vs_movement(lag=)`). This is the real-file test of the
  origin check: restoring the zeros puts 15.5 % of samples at the origin.
- **hc31_anm190422.** A 1-D environment 4815 cm long (1928 bins of 2.5 cm),
  position in inches and time in microseconds (G-093); the readme's
  interneuron rule cannot work on its uint8 `ID`, and every unit is kept.
- **hc18_train261.** A 32,552 Hz spike clock (not 20 kHz), one LED in a
  768 × 576 image, subsession events in ms, a closed circuit run about 107
  times. `detect_laps` finds 1 (G-037); 341 of 2487 visited bins hold a
  single 25.6 ms frame (G-022).

## Matrix rows

Status per operation (`results/matrix.md`; column order as there):

| dataset | exp. | env | occ | rates | SI | SI vs shifts | place | grid | border | hd | decode | nwb rt | graph | segments | laps | turns | replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hc11_achilles | 8/8 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | ok | ok | – | ok | err G-015 | – | wk G-038 |
| hc3_ec013_669 | 5/6 | ok ⚠1 | ok | ok | ok | ok | ok | ok | err G-005 | ok | ok | ok ⚠1 | – | ok | err G-015 | – | – |
| hc2_ec013_527 | 6/6 | ok ⚠1 | ok | ok | ok | ok | ok | ok | err G-005 | ok | ok | ok ⚠1 | – | ok | ok | – | – |
| hc28_hpa | 6/6 | ok | ok | ok | ok | ok | ok | ok | err G-005 | – | ok | ok | – | ok | err G-015 | – | – |
| hc31_anm190422 | 5/5 | ok | ok | ok | ok | ok | ok | – | – | – | ok | ok | – | ok | err G-015 | – | – |
| hc18_train261 | 5/5 | ok ⚠1 | ok | ok | ok | ok | ok | ok | err G-005 | – | ok | ok ⚠1 | – | – | ok | – | – |

`io.nwb_readers` and the imaging columns are `–` for all six. The ⚠ on the
pixel recordings is the units-registry warning (G-095). Decoding medians:
hc-11 8.5 cm (chance 38.8), hc-28 16.0 cm (42.1), hc-31 573 cm on a 48 m
track (1487), hc-3 12.2 px (114), hc-2 83 px (98), hc-18 303 px (440).
Units beating all four circular shifts: 98 % (hc-11), 93 % (hc-28), 92 %
(hc-31), 90 % (hc-3), 71 % (hc-2), 50 % (hc-18). hc-2's decoding is near
chance (83 px against 98): its 3210 bins of 2.5 px (about 2.5 cm) share
824 s of occupancy (median 0.18 s per visited bin), so each map rests on
little time per bin, decoded from 37 units in 0.25 s steps; larger bins were
not tried.

## hc-11 against DANDI 000044 (same session)

`studies/hc11_vs_000044.py`, output in `results/studies/hc11_vs_000044.md`:

| Quantity | 000044 (NWB) | hc-11 (sessInfo.mat) | Agreement |
| --- | --- | --- | --- |
| units | 137 | 137 | 137 paired by equal spike count, every spike time identical (largest difference 0 s), same order |
| spikes | 8,414,607 | 8,414,607 | |
| cell type | excitatory / inhibitory | pyramidal / interneuron | 137 of 137 agree |
| position samples | 80,762 | 80,762 | times differ by at most 5.0e-8 s |
| 2-D position (cm) | | | same NaN pattern, largest difference 0 cm |
| linearized (cm) | | | same NaN pattern, largest difference 0 cm |
| PRE / MAZE / POST (s) | 0–18079.5 / –20147.0 / –34861.1032 | same | identical |
| sleep states | awake 62 (16,747 s), non-REM 60 (15,890 s), REM 22 (2,071 s) | wake 20 (5,698 s), drowsy 72 (9,967 s), NREM 57 (15,488 s), intermediate 31 (510 s), REM 28 (2,653 s) | different scoring, not a merge |
| POST non-REM bursts (same rule) | 1646 | 1803 | differ through the sleep epochs |
| driver operations | | | equal to about 1e-11, except replay: `GROSMARK_2016` 14 % vs 16 % of 200 candidates |

The 000044 unit ids (1–272) are not the hc-11 cluster ids (shank × 100 +
cluster). Gap G-090.

## New gaps

- **G-090** (data-quirk): hc-11 and 000044 agree exactly except their sleep
  scoring, which changes the replay candidates.
- **G-091** (data-quirk): wrong expectations: hc-3 unit count (72 fire of 125
  sorted that day); hc-28 epoch read from the day-2 task file.
- **G-092** (data-quirk): hc-28's LED angle sits about −15° to −20° off the
  direction of movement (−14.4°, spread 18.8°, over 15 frames), outside the
  heading check at every lag; heading dropped rather than rotated.
- **G-093** (data-quirk): hc-31 in inches and microseconds, stated only by
  two scalars; its uint8 `ID` stores NaN as 0, so the readme's interneuron
  marker, if set, reads as "silent"; every unit is kept.
- **G-094** (silent-hazard): two LEDs at one point give heading exactly 0
  (`arctan2(0, 0)`); 5.1 % (hc-3) and 2.3 % (hc-2) of frames; nothing warns.
- **G-095** (false-alarm): `Environment` warns on `units="pixel"`, which
  `_units.py` aliases to `px`; two unit tables disagree.
- **G-096** (silent-hazard): the Neuroscope `-1` missing marker kept as a
  coordinate passes silently; the origin check cannot see it.
- **G-097** (silent-hazard): spikes on the wrong clock (20 kHz assumed for a
  32,552 Hz file) pass silently when the spikes that leave tracking fall
  outside the analysis epochs.
- **G-098** (harness): `heading_vs_movement` with one-sample steps is biased
  on whole-pixel positions (hc-28: spread 51° at one frame, 18.8° at 15);
  it gains `lag=` (default 1, unchanged).

## Warning calibration

Clean recordings, `studies/warning_calibration.py` on the six keys
(`results/studies/` restored afterwards; these are the numbers):

| Dataset | Origin share | Linearized origin share | Fewest bins | Heading bins | Heading bins, running |
| --- | --- | --- | --- | --- | --- |
| hc11_achilles | 0.000 % | 0.010 % | 81 | – | – |
| hc3_ec013_669 | 0.000 % | – | 66 | 30 | 31* |
| hc2_ec013_527 | 0.000 % | – | 73 | 36 | 36* |
| hc28_hpa | 0.000 % | – | 36 | – | – |
| hc31_anm190422 | 0.000 % (1-D) | – | 1928 | – | – |
| hc18_train261 | 0.000 % | – | 114 | – | – |

\* The study computes running coverage only for cm; these two use the
loaders' inferred cm/px (running above 10 cm/s), computed separately.

- **Check 1 (origin share ≥ 0.1 %, 2-D/3-D).** Fires on none of the six
  clean recordings (19 of 19 clean recordings with positions now). On the
  real file defect, hc-28's `rawpos` zeros restored, the origin share is
  15.5 %: check 1 fires. It would also catch every `missing_positions_as_zero`
  probe here (21 silent rows, all 2-D). It does not catch the Neuroscope `-1`
  marker (13 silent rows, G-096). hc-11's linearized coordinate has 0.010 %
  of samples exactly at 0 (the same as 000044), a real place.
- **Check 2 (fewer than 5 bins).** The fewest on a clean recording here is
  36 (hc-28). It catches ÷100 on hc-11 (3 bins) and hc-28 (4 bins); on the
  48 m track ÷100 leaves 21 bins, so only the speed gate refuses it (G-014).
- **Check 3 (heading coverage below 18 of 36 bins).** The hc-3 linear track,
  a second linear track with tracked head direction, fills 30 of 36 bins
  (31 while running): lower than every other recording (36) and still well
  above 18. hc-2's open field fills 36.

Hazard probes on the six recordings (73 rows; probes on operations that
already failed clean are `not probed` and not counted):

| Defect put back | Probes | Caught | Silent | No effect | Undetectable |
| --- | --- | --- | --- | --- | --- |
| Missing positions stored as 0 | 21 | 0 | 21 | 0 | 0 |
| `-1` missing marker kept (`.whl`, `.pos`) | 13 | 0 | 13 | 0 | 0 |
| Length unit off by 100 | 12 | 8 | 3 | 0 | 1 |
| Spike times in ms | 8 | 8 | 0 | 0 | 0 |
| Inches read as cm (hc-31) | 4 | 0 | 0 | 0 | 4 |
| Times in microseconds (hc-31) | 4 | 3 | 0 | 1 | 0 |
| MATLAB intervals left transposed (hc-11) | 4 | 2 | 0 | 2 | 0 |
| Headings in degrees | 2 | 2 | 0 | 0 | 0 |
| Coincident LEDs as east | 2 | 0 | 2 | 0 | 0 |
| LEDs swapped (heading + π) | 2 | 0 | 0 | 0 | 2 |
| 20 kHz clock assumed (hc-18) | 1 | 0 | 1 | 0 | 0 |
| **Total** | **73** | **23** | **40** | **3** | **7** |

Three `spike_times_in_ms` rows (partly) and the `clock_assumed_20khz` row
(only) read `warned` in `warnings.md` because of the pixel-units warning (G-095):
the clean run records it once, in `env.from_samples`, and reuses that
environment, while a probe builds its own environment inside the probed
operation, so the warning is new there. The verdicts judge the real
response.

## What should change in FINDINGS.md

- **Decision 4, check 1.** Add the real-file test: hc-28's documented
  `rawpos` zeros, 15.5 % of a W-track session at the origin, caught by check 1;
  19 of 19 clean recordings stay under the threshold. Note that check 1 does
  not cover non-zero markers (G-096).
- **Decision 1, heading coverage.** A second linear track: 30/36 bins (31
  running) on hc-3, the lowest so far, still above the threshold.
- **Not aggressive enough table.** Add the 73 rows above; new silent
  defects: the `-1` marker (G-096), coincident LEDs (G-094), a wrong spike
  clock with epochs (G-097).
- **Too aggressive.** One new false alarm on clean data: the `units="pixel"`
  registry warning (G-095), on every pixel recording.
- **Proposal 2 (laps, G-037).** A second real loop: the model train's circuit,
  about 107 laps, `detect_laps` finds 1.
- **Proposal 4 (G-022).** Single-frame bins on a fast-moving trajectory make
  the median peak rate exactly 1 spike / 1 frame (hc-18).
- **Decision 5 (laps in open arenas).** hc-2's open field gives 1 clockwise
  lap; the hc-28 W-track is refused, while 000447's W-track gave 1.
- **Data quirks.** Add G-090, G-092, G-093; the hc-11/000044 sleep scoring
  difference matters to anyone comparing replay rates across the archives.
- **Next batch / out of reach.** hc-13 is reachable by streaming (8.1 GB,
  about 11 min at the rate hc-18 achieved) if a second `rawpos` file is
  wanted; hc-6 not loaded.

## Not done or not verified

- The image y direction of `.whl`/`.pos`/`rawpos` is not stated; positions
  and headings stay in the stored frame and turn directions are not
  interpreted. No graph was built for the W-track or the train circuit, so
  `env.from_graph` and `behavior.turn_sequence` are `–`.
- The pixel scales are inferred from apparatus sizes and not checked against
  any stated scale; they set only bin sizes and speed thresholds.
- hc-18's train speed and circuit length are not in the documentation, so its
  bin size (5 px) and the absence of a speed threshold are choices, not
  facts; a larger bin would remove most single-frame bins (G-022).
- The hc-11 cross-check uses the sessInfo file, not the `.res/.clu` in the
  1.8 GB session archive.
