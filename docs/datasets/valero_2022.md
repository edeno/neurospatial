# Valero et al. (2022): mouse CA1 on a linear track, filmed from above

**The question.** How do hippocampal CA1 neurons respond to optogenetic pulses
while a mouse runs on a linear track? This dataset has 28 sessions with sorted
units, ripples, sleep states and an overhead video of each session. It is the
dataset to use for drawing tracked position and spikes over the camera's view
of the animal.

!!! warning "Baseline and stimulation runs are mixed"
    Each track session interleaves "three blocks of 10 baseline runs" with
    "two blocks of 40 to 50 stimulation runs" (main text), and light unmasks
    place fields in non-place cells, the paper's own finding. The stimulus
    group is empty in 26 of the 28 files (checked on all of them), so they
    hold no stimulation times; the other two (`sub-fCamk1_ses-fCamk1-200827-sess9`
    and `sub-fCamk1_ses-fCamk1-200901-sess12`) store optogenetic series that
    the loader does not read. The summary does not separate the blocks. Place
    fields and rate maps computed on the track epoch mix the two conditions.
    Every session carries this as a watch-out.

| | |
| --- | --- |
| Dataset id | `valero_2022` |
| Animal and region | CaMKIIa-Cre x Ai32 mice (channelrhodopsin in CaMKIIa neurons), CA1 |
| Apparatus | 110 cm x 6.35 cm PVC linear track, filmed from above |
| Archive | DANDI 000568, version 0.230705.1633 (CC-BY-4.0), the files without raw voltage (`-no-raw-data`), 5.8 to 53 MB each; the copies with raw voltage are 1 to 54 GB |
| Sessions | 28. The default, `sub-fCamk2_ses-fCamk2-201014-sess3` (35 units), is checked by hand |
| Video | 27 of the 28 sessions have an `.avi` beside them on DANDI (46 MB for the default), fetched only on request; needs the `video` extra. 14 sessions have two video files, which gives `rec.video` of `None`, and `sub-fCamk5_ses-fCamk5-210406-sess10` has none |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'valero_2022'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

!!! warning "The video overlay is aligned with the tracking by construction"
    The files state no pixel size and store no pixel tracking, so the loader
    calibrates the video by fitting the LED it detects in the frames to the
    file's positions. The overlay shows what the camera saw at each moment. It
    is not a check of the tracking or of the declared track, and the track is
    not visible in the dark video. Only the default session has a calibration.

<!-- docs-test: skip needs a download -->
```python
from neurospatial.datasets import load

rec = load("valero_2022")
print(rec.video)                          # where the video is; nothing is downloaded yet
start = rec.times[0]
scene = rec.animation(time_range=(start, start + 20), video=True)   # fetches the video
scene.export("valero_video.mp4", fps=30)
```

## What was recorded

CA1 units of CaMKIIa-Cre x Ai32 mice probed with optogenetic pulses, over
sessions of five epochs on one clock: one epoch on a PVC linear track "110 cm
long, 6.35 cm wide" (the description of each file's position series), the others
off it. Position is one tracked point from an overhead Basler camera, in cm,
every 1/30 s. The files also hold ripples, sleep states (wake, NREM, REM) and
up/down states.

Sessions are DANDI 000568's files without the raw voltage, named without the
`-no-raw-data` suffix. The default session, `sub-fCamk2_ses-fCamk2-201014-sess3`,
has 35 units and the track from 7,674 to 9,941 s.

Sources read: DANDI 000568's metadata, the files, the paper's Crossref record
(title, authors, volume, pages) and the paper's main text (the author
manuscript, PubMed Central PMC9632609). Its Materials and methods, in the
Supplementary Materials, were not available to read, so no number on this page
comes from the paper.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: the file's `SpatialSeriesRaw` as stored, in cm | stated | The file's unit. The head is tracked, so samples spread beyond the 6.35 cm track width. |
| times: the position timestamps; the video's frame times follow them | inferred | The video `ImageSeries` holds the same frames (as many as position samples) on a clock 22.55 to 23.12 s earlier. The spikes are on the position clock, so the overlay follows it. |
| epochs: the file's epochs, those holding the tracking as `track` (two in sessions that run the track twice), the others as `epoch_<k>`; the stretches of a track epoch without positions as `track_before_tracking` and `track_after_tracking`; `SleepStates` as `wake`, `nrem`, `rem` | stated | The file's epochs table, position timestamps and SleepStates labels. |
| events: `ripples`, the `RippleTimeIntervals` rows, [start, stop) in s | stated | The file. |
| `units["brain_region"]`: CA1 for every unit | inferred | The electrodes table gives the locations; the units table names no electrode or location. |
| environment: the stated 110 cm track as a one-edge graph along the samples' principal axis, between their 0.5th and 99.5th percentiles, 2.5 cm bins | inferred | The file states no origin or rotation; its reference frame is "Arbitrary, camera". |
| environment: `track_width`, a placement tolerance, not the stated 6.35 cm | inferred | Twice the 99th percentile distance of the samples from the track. It decides only which samples are placed, not the bins. |
| video calibration, default session: an affine from the LED detected in the video to the file's positions | inferred | See "The video" below. |
| headings | none | One point is tracked. |

## The video

`rec.video` is a `VideoSource`: the file, one time per frame, a calibration and
the basis of each. Nothing is downloaded until `rec.video.fetch()` or
`rec.animation(video=True)`.

**The overlay is aligned with the tracking by construction.** The file states
no pixel size and stores no pixel tracking, so the default session's video is
calibrated by a least-squares affine from the LED detected in the video to the
file's positions (`scripts/fit_video_led_calibration.py`, run once and stored
in the registry). The overlay therefore shows the camera's view of each moment;
it is not a check of the tracking or of the declared track, which the dark
video does not show. A watch-out repeats this, and `rec.video.calibration_basis`
gives the fit and its residual.

Other sessions refuse `rec.animation(video=True)` until the fit is run for
them; the `InputError` names each calibration source tried. A session whose
video is split across two files (14 of the 28) has `rec.video` of `None`.

The scene of the first 20 s over the video is drawn on a grid over the track's
footprint (`scene.env`), with the track as its along-track panel. To put your
own decoded posterior into the same scene, decode on `scene.env` (a 2-D grid
over the box around the track), not on the track graph `rec.env`: a posterior
over the track's bins lies along the track and cannot be drawn over the video's
2-D frames (draw it with `rec.animation()` instead, without the video).

## Watch out

For every session: the baseline and stimulation runs are mixed, as in the box
at the top of this page.

For the default session `sub-fCamk2_ses-fCamk2-201014-sess3`:

- **The recording date.** The file's `session_start_time`, 2016-03-15, is not
  the recording date: the epoch names (`fCamk2_201014_091141` to
  `fCamk2_201014_122743`) give 2020-10-14. Times are seconds from the first
  epoch's start.
- **Spikes without positions.** The track epoch starts 22.6 s before the first
  position sample (7,674.30 s against 7,696.89 s), so the first 22.6 s of the
  epoch has spikes and no positions (`epochs["track_before_tracking"]`); the
  file does not say why.
- **The spikes are on the position clock, not the video's.** The 35 units'
  spatial information (`compute_spatial_rates` with `method="binned"` on this
  environment, the whole session) averages 0.84 bits/spike at the position
  timestamps, 0.36 with the positions moved 22.84 s earlier onto the video's
  clock, and 0.30 to 0.43 with them moved 10 or 40 s either way (checked on
  2026-11-03).
- **Sleep scoring inside the track epoch.** The file's SleepStates score NREM
  during parts of the track epoch (for example 9,723 to 9,939 s, while the
  mouse runs along the track at 9,914 s): select running by speed, not by
  `epochs["wake"]`.

For every session: the file also holds up and down states
(`UpDownStatesTimeIntervals`) and the ripples' peak times, which are not
loaded; read them with pynwb. A share of tracked samples lies farther than half the stated
track width from the track (the head is tracked, not the body), and the
summary gives the share. For a session with video, further watch-outs give the
clock offset between the video and the positions, the LED fit's residuals (and
the share of detections left out), and that the video is shown mirrored from the
raw file, since the file's y grows down the camera's rows.

One session, `sub-fCamk2_ses-fCamk2-201013-sess2`, has an epoch that ends 50 s
after the recording, so it loads only with `accept_failed_checks=("clock",)`.

## Analysis choices

`rec.params`: `bin_size` 2.5 cm, the loader's choice (no source read states
one); `min_speed` `None` (no source read states a running threshold); `dt`
0.25 s, the decoding bin; `train_epochs` and `test_epochs`, alternating 60 s
blocks of the tracked span.

## Worked result

**Status: not possible yet.**

The finding to reproduce: most CA1 pyramidal neurons on the track are place
cells. The paper: "With a standard definition of "place field" (materials and
methods) (18), more than half of the pyramidal neurons were classified as place
cells [73% and 71% of light-responsive and nonresponsive neurons, respectively
...]" (main text, the paragraph on place cells; fig. S11).

Three things are missing:

1. The place-field definition is in the paper's Materials and methods, in the
   Supplementary Materials, which were not read.
2. The share is of pyramidal neurons, and the file's units table holds no cell
   type (`clu_id` and `unit_name` only).
3. Each track session interleaves "three blocks of 10 baseline runs" with "two
   blocks of 40 to 50 stimulation runs" (main text), where light unmasks place
   fields in non-place cells, the paper's own finding. The default file holds
   no stimulation times (its stimulus group is empty, as in 26 of the 28
   files), so its baseline runs cannot be told apart.

Source: Valero et al. 2022, Science 375, 570-574, main text (author
manuscript, PMC9632609).

## Cite

Valero M, Zutshi I, Yoon E, Buzsáki G (2022). Probing subthreshold dynamics of
hippocampal neurons by pulsed optogenetics. Science 375, 570-574.
<https://doi.org/10.1126/science.abm1891>

Data: Valero, Manuel; Zutshi, Ipshita; Yoon, Euisik; Buzsáki, György (2023)
Probing subthreshold dynamics of hippocampal neurons by pulsed optogenetics
(Version 0.230705.1633) [Data set]. DANDI archive.
<https://doi.org/10.48324/dandi.000568/0.230705.1633>
