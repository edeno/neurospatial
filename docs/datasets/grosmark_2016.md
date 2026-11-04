# Grosmark and Buzsáki (2016): rat CA1 on a novel maze, with sleep before and after

**The question.** How do hippocampal place fields form on a maze the rat has
never seen, and how does sleep before and after relate to them? A session is
about ten hours: sleep in the home cage (PRE), running for water on a novel
maze (MAZE), and sleep again (POST). This dataset suits place fields on a
linear track, circular-track coordinates, and sleep-state analyses.

| | |
| --- | --- |
| Dataset id | `grosmark_2016` |
| Animal and region | four rats, dorsal hippocampus CA1, silicon probes implanted bilaterally |
| Apparatus | one novel maze per session: a 1.9 m wooden linear maze (1.6 m stem), a 2.3 m metal linear maze (2 m stem) or a 1 m diameter wooden circular maze |
| Archives | OSF `hu5ma` (765 MB; license not stated) for the default session; DANDI 000044, version 0.261005.2101 (CC-BY-4.0), for eight more |
| Sessions | 9. The default, `Achilles_10252013`, is checked by hand; the eight DANDI sessions are not yet |
| Size | the DANDI files are 5 to 9 GB each, most of it LFP; load them with `stream=True` |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

catalog().query("dataset == 'grosmark_2016'")[["session", "n_units", "size_bytes", "verified", "default"]]
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial.datasets import load

rec = load("grosmark_2016")                       # the OSF file, 765 MB
rec = load("grosmark_2016", "sub-Achilles_ses-Achilles-10252013", stream=True)
```

## What was recorded

Units in dorsal CA1 of four rats. The Supplementary Materials' Methods give the
mazes: a 1.9 m wooden linear maze with a 1.6 m stem and a 15 cm reward area at
each end (4 sessions); a 2.3 m metal linear maze with a 2 m stem and the same
reward areas (1 session); and a 1 m diameter wooden circular maze, run
clockwise, with a 30 cm reward area (3 sessions). Two LEDs 5 cm apart on the
head-mounted Faraday cage were tracked "at 39.69 Hz"; the files sample every
0.0256 s (39.06 Hz).

A session is one archive file, so one recording on one maze.

- **The default session**, `Achilles_10252013`, is the NWB file that NeMoS's
  place-cell tutorial downloads: OSF `hu5ma` in the "NeMoS Tutorial" project.
  It holds 137 units (120 putative pyramidal `pE`, 17 interneurons `pI`), 84
  trials, the theta phase at 1250 Hz, and a 1-D position along the 1.6 m linear
  maze's stem.
- **The other eight sessions** are the files of DANDI 000044, one per maze
  session, among them `sub-Achilles_ses-Achilles-10252013`, DANDI's copy of the
  default session. They hold 2-D positions, the authors' linearized position,
  the PRE/MAZE/POST epochs and a sleep scoring.

Sources read: the paper and its Supplementary Materials (Science 351,
1440-1443), NeMoS's place-cell tutorial, the OSF file's metadata, DANDI
000044's metadata, and the nine files.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| OSF positions: `acquisition/position` as stored, in cm, one column | stated | The file's unit. |
| DANDI positions and linearized positions: metres times 100, cm | stated | The files' unit and conversion. Untracked samples are NaN in the files and stay NaN. |
| OSF epochs: `pre`, `maze` and `post` are DANDI's copy of the same session, on the same clock | inferred | The OSF file has no epochs. |
| OSF epochs: `maze_before_track` from the start of the maze epoch to the file's first position sample | inferred | DANDI's copy shows the rat off the track there. |
| DANDI epochs: `PREEpoch`, `MazeEpoch`, `POSTEpoch` as `pre`, `maze`, `post`; the `states` table as `awake`, `non_rem`, `rem` | stated | The files' tables. |
| environment, OSF: the stated 1.6 m stem as a 1-D track at the samples' midrange, 2 cm bins | inferred | The file states no origin for its position. The reward areas are left out, as the file's samples are. |
| environment, DANDI linear maze: the stated maze (15 cm reward area, stem, 15 cm reward area) as a straight track along the line the linearized position traces in 2-D | stated apparatus, fitted placement | A least-squares fit from the linearized position to the positions, the stem centred on it. |
| environment, DANDI circular maze: the stated 1 m circle as a closed loop of 158 segments of 1.99 cm, centred by a robust circle fit | stated apparatus, fitted placement | `env.to_linear` runs counter-clockwise from east of the centre, in the file's coordinates. |
| environment: `track_width`, a placement tolerance | inferred | Twice a high quantile of the tracked samples' distance from the track, and at least one bin. The 99th percentile on a circle; the 95th on a linear maze, where the rat spends the first minutes off the track (Achilles 10252013: about 110 s, 3.5% of samples) and the 99th would make the track 3 m wide. The OSF 1-D stem has none: its positions lie along the track by construction. |
| headings | none | `rec.headings` is `None`: the files hold one position per sample, not the two LEDs. |

## Watch out

For `Achilles_10252013` (OSF):

- **The position is 1-D.** It is the head's distance along the linear maze's
  axis, in cm, for the samples on or near the track. Against DANDI's copy it is
  `33.10 + 0.99975 x - 0.02299 y` of the 2-D positions at the same times
  (residual SD 0.04 cm, at most 0.07 cm, over 12,817 samples): a projection
  onto the track's axis, 1 cm per cm. The rat's time elsewhere (the start of
  the maze epoch, the reward ends beyond the samples' range, stretches far
  beside the stem) is left out, not stored as NaN.
- **It is not DANDI's linearized position.** Where both are defined (10,336
  samples) it is 8.17 + 0.9626 times DANDI's (residual SD 0.11 cm): DANDI's
  0-160 cm covers 154.2 cm of this axis, and the two differ by up to 8.4 cm.
  This file keeps 2511 samples where DANDI's is NaN. Place-field positions from
  the two archives are not interchangeable.
- **Some positions are interpolated.** 30 samples, in 5 runs of 2 to 19 samples
  (the longest from 19,136.42 s), lie on a straight line between their
  neighbours: DANDI's copy has no 2-D position there, so this file's values are
  interpolated, not tracked.
- **The file has no epochs and no sleep scoring.** `epochs["pre"]`, `["maze"]`
  and `["post"]` are DANDI's for this session. Load
  `sub-Achilles_ses-Achilles-10252013` for its sleep states.
- **Theta phase is not loaded.** The file also holds `acquisition/theta_phase`
  (1250 Hz, the whole session); read it with pynwb or pynapple. It is in
  radians, so it is not a position.
- **Trials.** `rec.trials` holds the file's 84 trials, described only as
  "experimental trials"; NeMoS's tutorial uses them as the times the rat
  traversed the track. The summary gives how many span the track and how long
  they last. Each starts and stops at a position sample; read as
  [start, stop), a trial leaves out its stop sample, which pynapple's inclusive
  intervals keep.
- **Tracking rate.** The paper states LED tracking at 39.69 Hz; the files
  sample every 0.0256 s, 39.06 Hz, 1.6% slower. The files' rate is used.

For the DANDI sessions: `rec.linearized` is the file's linearized position,
defined only on the stem (NaN at the reward ends and off the track). On a
linear maze it can be rescaled rather than distance along the track; a
watch-out gives the session's slope. On the circular maze it leaves out the
reward area, and in the three files it increases counter-clockwise while the
rat runs clockwise, so it decreases along a run. Each session's watch-outs give
its own numbers, and the reward ends and any uncovered stretch of the loop.

## Analysis choices

`rec.params`: `bin_size` 2 cm and `min_speed` 5 cm/s, from the Supplementary
Materials' place-field analysis ("restricted to epochs during which the
animal's velocity was at least 5 cm/s"). The paper also left out the reward
areas and analysed each running direction on a linear maze separately, which
`params` does not do. `dt` 0.25 s, the decoding bin, and `train_epochs` and
`test_epochs`, alternating 60 s blocks of the tracked span.

## Worked result

**Status: reproduced. Level B (a property, tested against a null).**

Finding: the rate-map peaks of each running direction are enriched near the
stem's two ends, beyond a shifted-spike null.

The paper: "Note that the place fields of rigid (blue) and plastic cells (red)
are similarly enriched near the start and goal ends of the track." (Fig. S2
legend). The place-field analysis used 2 cm bins, spikes and occupancy each
smoothed with a 5 cm Gaussian, velocity at least 5 cm/s, and "For linear track
maze runs all place field analysis was carried out independently for left and
right directions of movement" ("Place Field Analysis"). The paper states no
share; it reports 491 place fields over 8 sessions.

| | Value |
| --- | --- |
| Ours | 50.3% of directional rate-map peaks lie within 20 cm of an end of the stem |
| Null | 22.2% to 36.5% with each unit's spikes shifted circularly (20 draws, seed 2016), 95th percentile 36.3% |
| Criterion (fixed before the run) | the share exceeds the null's 95th percentile. A fixed 25% bar is not used: shifted spikes alone give 22 to 36%, since occupancy, speed and smoothing crowd peaks at the ends |

How it was computed: the 120 pyramidal (`pE`) units; per direction (the sign
of each step's displacement) `compute_spatial_rates` with `method="binned"`,
bandwidth 5 cm and `min_speed` 5 cm/s, on `rec.env`'s 2 cm bins; the maps
peaking at 1 Hz or more, both directions pooled. The null uses
`stats.shuffle_spike_times_circular` on runs without a gap over 1 s, with
shifts of at least 20 s.

Not matched to the paper:

- the place-cell test: the paper keeps place cells by its occupancy resampling
  null (5,000 draws, 5 consecutive bins above the 99th percentile); this keeps
  every pyramidal map peaking at 1 Hz or more, so the real and null shares come
  from the same selection;
- the reward areas: the paper leaves them out; the OSF file's positions cover
  the stem only, so nothing is left out;
- the 5 cm Gaussian is read as its standard deviation.

Source: Grosmark & Buzsáki 2016, Supplementary Materials: Methods, "Place Field
Analysis" (p. 4), and the Fig. S2 legend (p. 14).

## Cite

Grosmark AD, Buzsáki G (2016). Diversity in neural firing dynamics supports
both rigid and learned hippocampal sequences. Science 351, 1440-1443.
<https://doi.org/10.1126/science.aad1935>

Data (DANDI sessions): Grosmark, Andres D.; Buzsáki, György (2026) Diversity in
neural firing dynamics supports both rigid and learned hippocampal sequences
(Version 0.261005.2101) [Data set]. DANDI Archive.
<https://doi.org/10.48324/dandi.000044/0.261005.2101>

Data (default session, OSF): Balzani, Edoardo; Broderick, William F; Viejo,
Guillaume (2024) NeMoS Tutorial: datasets for the NeMoS tutorials. OSF.
<https://osf.io/5crqj/> (the file `hu5ma` states no license).

`rec.citation` holds the paper's citation and the citation of the session's own
archive.
