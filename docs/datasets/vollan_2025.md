# Vollan et al. (2025): rat grid modules in open fields and on linear tracks

**The question.** Do grid cells of different modules share a map, and how do
their scores compare across modules of different spacing? This dataset holds
hundreds of simultaneously recorded medial entorhinal cortex units per rat,
already sorted by the authors into grid modules with their grid scores, and
head direction tracked with position. It suits grid scores per module,
head-direction and conjunctive tuning (heading is checked against movement)
and linear-track place and grid fields.

| | |
| --- | --- |
| Dataset id | `vollan_2025` |
| Animal and region | Long Evans rats; medial entorhinal cortex-parasubiculum and hippocampus |
| Apparatus | 150 x 150 cm open-field box, a 150 cm circular arena, or a 200 cm linear track |
| Archive | EBRAINS doi:10.25493/R5FR-EDG, v1 (CC-BY-SA-4.0); a public bucket, no login; one MATLAB file of 101 to 852 MB per session |
| Sessions | 38: every open-field (`of_<rat>_<recording>`, 31) and linear-track (`lt_<rat>_<recording>`, 7) file. Two are checked by hand: the default `of_26018_2` (644 MEC units, three grid modules) and `lt_27764_1` (389 MEC units) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'vollan_2025'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("vollan_2025")
rates = compute_spatial_rates(
    rec.env,
    rec.spike_times,
    rec.times,
    rec.positions,
    max_gap=rec.params["max_gap"],  # do not bridge the speed filter's gaps
)
module_1 = rec.units["module"] == 1
```

## What was recorded

Neuropixels units in the medial entorhinal cortex-parasubiculum region (MEC)
and the hippocampus of Long Evans rats that foraged in an open field or
shuttled between rewards on a 200 cm linear track. Head position and head
direction come from OptiTrack motion capture of markers on the implant,
projected onto the horizontal plane, every 0.01 s (100 Hz). The authors
grouped MEC units into grid modules on each rat's open-field session; each
file stores the modules, their units and their grid scores.

The default session, `of_26018_2`, is rat 26018's lit open-field session: 644
MEC units, of which the authors' three grid modules hold 48, 90 and 95, with
median unit grid scores 1.114, 0.986 and 0.685 and spacings of 46, 65 and
93 cm. `lt_27764_1` is rat 27764 on the linear track: 389 MEC units, 84 of them
in the rat's one grid module.

`rec.units` gives each unit's area (`brain_region`), the authors' functional
class (`label`: for example `nongrid`, `bursty`, `conjunctive`, `id`), the
Kilosort label (`ks2_label`, `good` or `mua`, both kept as shared), its grid
module (`module`, 0 for none) and `is_grid`. `rec.params` repeats the module
membership and gives each module's grid scores and spacing.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: `x`, `y` in m, times 100 to cm | stated | The dataset's `mat_files.md`: "head x-position relative to arena center (m)". |
| headings: `hd` kept as radians in [-π, π], allocentric in the positions' frame | inferred | `mat_files.md` says only "2D head direction (azimuth)". Minus the direction of movement while running, its circular mean is within 6 degrees of 0 in every file. |
| times: `t` as stored, with gaps | stated | "All timeseries data has been 'speed-filtered' to discard samples when the animal's locomotion speed was below 5 cm/s" (`mat_files.md`). The gaps are kept; rebuilding times from `dt` would move every later sample. |
| units: every area, both Kilosort labels; `cellType`, `acorrCluId` and `isGrid` as `label`, `module` and `is_grid` | stated | `mat_files.md` describes each field; each module's `unitIds` list the units whose `acorrCluId` is its index. |
| epochs: each listed session's `validTimeRanges`, under its name | stated | `mat_files.md`: sessions, "list of individual sessions (time epochs) included". |
| epochs: rows past the recording (its first to last sample or spike) clipped to it, rows wholly outside it dropped | inferred | The Methods truncate awake sessions "to the nearest multiple of 100 s ... by trimming the last part of the behaviour session", so a listed session can end after the data. Every clipped or dropped row is in the `Conversion` and under "Watch out". An edge within one sampling interval (0.01 s) of the recording is kept as stored. |
| environment: the box, the circle or the track, placed on the samples | inferred | The Methods state the sizes; the files state no placement. The box sits on the samples' minimum rotated rectangle, the circle on their minimum bounding circle, the track along their principal axis. |

The arena follows the file's session type. `open_field` is the Methods'
"square open-field box with a floor size of 150 x 150 cm". `open_field_light`
and `open_field_darkness` are taken to be the "circular arena 150 cm in
diameter" the Methods use for a dark-room session and the lit session after
it: the paper names no rat or file, so this is inferred from the session type
and from the samples, which are round in those files and square in the others.
The `Conversion` gives how much of a rectangle and of a circle the samples
fill. `linear_track` is the "200 cm linear track"; no width is stated, so the
track is built with a placement tolerance measured from the samples.

The sampling rate (100 Hz) is stated: `dt` is "0.0100 sampling time interval
(s)". The units check refuses a session whose largest span lies outside 120 to
240 cm, 20% below the 150 cm arenas to 20% above the 200 cm track.

## Watch out

- **Speed-filtered times.** The authors removed every sample slower than
  5 cm/s, so `rec.times` has gaps: 173 longer than 0.02 s in the default
  session, the longest 253 s. `compute_spatial_rates` with its default
  `max_gap=1.0` bridges the short gaps as occupancy and warns that the times
  look boolean-masked. Pass `max_gap=rec.params["max_gap"]` (0.015 s) and no
  `min_speed`.
- **Session tables that disagree with the samples.** Awake sessions were cut
  to a whole number of 100 s, so the default session's samples end 352 s
  before its listed end. In six files samples run past the listed session
  (55% of `lt_27764_1`'s samples come after it, up to 8412 s), and in two a listed
  session holds no samples at all (`of_25843_5`, `of_26820_2`: the darkness
  sessions are listed before the samples begin). Epochs are clipped to the
  recording, and a listed session wholly outside it gives no epoch; the
  summary gives every clipped row with its numbers. No sample or spike is
  dropped.
- **The circular arena is inferred.** See above. For the default session the
  samples fill 81% of their minimum rectangle and 94% of their minimum circle.
- **Arena centres.** The files give positions "relative to arena center", but
  in every file the origin is the samples' midrange. Where far samples move
  the midrange the two differ: in `of_26034_3` the box placed on the samples
  sits 9.3 cm from the file's (0, 0). The summary reports a distance over
  5 cm.
- **Modules on the track.** A linear-track file's modules and their grid
  scores are from the rat's open-field session (`unitAcorrClus.sessionType`
  `'of'`), not from the track.
- **Hippocampus-only rats.** `of_24666_1` and `of_25954_1` hold only
  hippocampal units and no grid modules or cell classes.
- **Tracking.** `lt_27764_1` has 18 runs of 11 or more identical position
  samples and 13 samples faster than the stated speed limit from the last
  believable one. They are kept.

## Analysis choices

`rec.params`: `bin_size` 2.5 cm, the Methods' rate-map bin; `min_speed` `None`
(the data are already speed-filtered at 5 cm/s); `max_gap` 0.015 s, one and a
half sampling intervals, so a speed-filter gap is not counted as occupancy;
`dt` 0.25 s and `train_epochs` / `test_epochs`, alternating 60 s blocks, for
held-out decoding; the authors' `module`, `is_grid` and `ks2_labels` per unit;
and per module (`module_ids`) `module_median_grid_score`, `module_grid_score`
(the grid score of the module's median autocorrelogram) and
`module_spacing_cm`.

## Worked result

**Status: not possible yet.** The comparison was attempted and is recorded.

The file stores each grid module's median unit grid score from the authors'
own maps and scorer (2.5 cm bins, each map smoothed by a cross-validated
Gaussian kernel, Methods). On the default session, neurospatial's grid score
(`compute_spatial_rates` on `rec.env` with `method="binned"` and a 5 cm
bandwidth, `max_gap=rec.params["max_gap"]`, then `grid_score` of
`spatial_autocorrelation`) gives:

| Module | Spacing (authors) | Authors' median | neurospatial median |
| --- | --- | --- | --- |
| 1 | 46 cm | 1.114 | 1.176 |
| 2 | 65 cm | 0.986 | 0.981 |
| 3 | 93 cm | 0.685 | 0.185 |

Modules 1 and 2 agree within 0.07; module 3, the largest spacing, does not.
neurospatial's autocorrelogram leaves undefined every lag where fewer than a
quarter of the visited bins overlap, and in a 150 cm arena the ring of peaks a
93 cm grid score reads lies at such lags. That rule is an open question for
the grid-score code; this dataset records the disagreement and does not change
the score. The authors' per-map smoothing and their scorer (their reference
37, not read) also differ from ours.

## Cite

Vollan AZ, Gardner RJ, Moser M-B, Moser EI (2025). Left-right-alternating
theta sweeps in entorhinal-hippocampal maps of space. Nature 639, 995.
<https://doi.org/10.1038/s41586-024-08527-1>

Data: Vollan, Abraham Zelalem; Gardner, Richard; Moser, May-Britt; Moser,
Edvard (2024). Left-right-alternating theta sweeps in the entorhinal-hippocampal
spatial map (v1) [Data set]. EBRAINS. <https://doi.org/10.25493/R5FR-EDG>
