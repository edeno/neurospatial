# Kanter et al. (2025): rat entorhinal cortex and hippocampus in open fields, with novel objects

**The question.** How do lateral entorhinal cortex (LEC), medial entorhinal
cortex (MEC) and the hippocampus represent an open field, and what changes
when a novel object appears in it? This dataset is Neuropixels recordings from
nine rats foraging in square open fields, some sessions with three novel
objects placed one after another. It suits spatial rate maps, head-direction
tuning with position, and comparing the empty arena with each object epoch.

| | |
| --- | --- |
| Dataset id | `kanter_2025` |
| Animal and region | Long Evans rats; LEC, MEC (layer 2) and hippocampus (CA1 or CA3) |
| Apparatus | Square open field of 1, 1.5 or 2 m with 50 cm walls, empty or with novel objects (the data descriptor) |
| Archive | EBRAINS 10.25493/GTVB-FQR (CC-BY-SA-4.0), a public bucket with no login; NWB files of 10 to 122 MB |
| Sessions | 31 foraging and novel-object sessions. One is checked by hand: the default `29629_2024-Oct-29_15-05-13` (380 units) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'kanter_2025'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("kanter_2025")
empty = compute_spatial_rates(
    rec.env, rec.spike_times, rec.times, rec.positions, epochs=rec.epochs["empty"]
)
first_object = compute_spatial_rates(
    rec.env, rec.spike_times, rec.times, rec.positions, epochs=rec.epochs["object_1"]
)
```

## What was recorded

Neuropixels 2.0 units in LEC, MEC and the hippocampus of nine adult male Long
Evans rats. In the foraging task the rat searched for scattered food crumbs in
a square arena. In the novel object task it foraged 7.5 min in the empty arena,
then a novel object (a footprint of about 10 x 10 cm, about 15 cm tall) was
placed at a pseudorandom location for 7.5 min, then a second and a third, each
somewhere else: the sequence Empty, OBJ1, OBJ2, OBJ3. Head position and head
direction come from five retroreflective markers on the implant, tracked by
OptiTrack motion capture and projected onto the floor, at 120 Hz.

The default session, `29629_2024-Oct-29_15-05-13`, is rat 29629 in the 1 m
arena with three objects: 156 LEC and 224 CA1 units, 125 of them labelled
`good` by the spike sorting.

Which files are offered: the bucket's `docs/index.json` lists each file's
tasks. The loader reads the 31 files whose epochs are all foraging or novel
objects (30 in the index, and `27963_2022-Feb-11_14-58-41`, which the index
leaves out). The figure-eight, odor-sequence and sleep files are not read.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: cm = the x and y columns of `animal_position` in m, times 100; the third column (head height) is not used | stated | The file stores cm with `unit="meters"` and `conversion=0.01`; the data descriptor says the markers were projected onto the horizontal plane. |
| headings: the file's head direction wrapped from [0, 2π) to [-π, π] | stated | `head_direction` is in radians with frame "(x=1, y=0) unit vector". |
| headings: in the positions' frame, 0 = +x and counter-clockwise | inferred | Compared with the direction of movement above 20 cm/s at about 10 Hz; the offset, its spread and how it drifts are in the `Conversion` and, where the offset is beyond 0.2 rad, in a watch-out. |
| unit ids: `<area>-<id>`, for example `LEC-12` and `CA1-12` | stated | The units table repeats ids across probes; each probe records one area. `units["file_unit_id"]` keeps the file's id. |
| epochs: `empty`, `object_1`, `object_2`, `object_3` for a novel-object session; `foraging` otherwise | stated | The data descriptor's sequence, and the index's task list for each file. |
| epochs: rows past the recording (its first to last sample or spike) clipped to it | inferred | Four sessions list an epoch ending 14.7 to 31.9 s after the last sample and spike. Each clipped row is in the `Conversion` and under "Watch out". |
| environment: the stated square arena (the epochs table's `arena_dimensions`), axes as in the file, placed where it holds the most tracked samples, 2.5 cm bins | inferred | The stated origin, "Bottom left of the arena", does not hold: positions run negative. |

The sampling rate (120 Hz) is **inferred** from the files' timestamps: no text
read states it. The stated position span covers the 1 to 2 m arenas; each
session is also compared with its own stated arena in a watch-out.

## Watch out

- **No object positions.** The files hold no object location, the data
  descriptor says only that each object was placed "at a pseudorandom
  location", and the dataset publishes no video. The epochs say when each
  object was in the arena, not where, so analyses that take
  `object_positions` cannot run.
- **Repeated unit ids.** The units table repeats ids across its probes (380
  rows and 235 distinct ids in the default session), so
  `neurospatial.io.nwb.read_units` refuses it, and keying units by the file's
  id would merge units of different brain areas. Use `rec.units.index`.
- **The stated origin does not hold.** The reference frame says "Bottom left
  of the arena", but most samples have a negative coordinate (in the default
  session x runs from -40.6 to 57.8 cm and y from -73.8 to 23.9 cm). The walls'
  positions are not stated, so the declared square is placed on the samples.
- **Head direction not verified by movement.** In 22 of the 26 sessions that
  load, heading minus the direction of movement has an offset beyond 0.2 rad.
  In the default session it is -0.33 rad with a median spread of 17 degrees
  (81 degrees with the heading negated, so it is not mirrored), and the means
  of 60 s windows run from -0.62 to +2.33 rad: the offset drifts, so no single
  correction follows from movement. The stored values are kept. Tuning-curve
  shapes do not depend on the frame; absolute preferred directions do. Two
  sessions of rat 26863 have offsets near ±π.
- **Epochs that end after the data.** In `27284_2021-Jun-16_14-33-11`,
  `27285_2021-Jun-30_13-59-11`, `27963_2022-Feb-08_21-14-27` and
  `27963_2022-Feb-10_11-06-56` an epoch ends 14.7 to 31.9 s after the last sample
  and spike. The epoch is clipped to the recording (its first to last sample
  or spike), and the summary gives the listed and the clipped row.
- **A session the clock check refuses.** In `26965_2020-Dec-08_11-29-48`
  tracking stops at 709 s while spikes and the epoch run to 906 s (the file's
  description says "motive crashed"): 20% of the spikes have no position.
  `accept_failed_checks=("clock",)` loads it anyway.
- **Sessions that do not match their stated arena.** In five sessions the
  samples' range is more than a bin past or 20% short of the arena the epochs
  table states, for example `27963_2022-Feb-11_14-58-41` (94 cm of samples in a
  stated 150 cm arena) and `27285_2021-Jul-02_13-23-12` (209 cm in 150 cm). The
  summary says so for each.
- **Unit quality.** Most units are labelled `mua` by the spike sorting; every
  unit is kept, and `units["cluster_quality"]` holds the label.
- **Epoch lengths.** In the default session the empty-arena epoch runs 605 s,
  not the descriptor's 7.5 min; the object epochs run 450, 523 and 452 s.

## Analysis choices

`rec.params`: `bin_size` 2.5 cm; `dt` 0.25 s, the decoding bin; `min_speed`
`None` (foraging, as for the other open fields); `train_epochs` and
`test_epochs`, alternating 60 s blocks, so decoding is always tested on
held-out time.

## Worked result

**Status: not possible yet.**

The data descriptor reports no result, and the associated article was not
read, so no published number can be reproduced from a source that was read.
The object results the novel-object sessions invite need the objects'
positions, which the archive does not hold.

## Cite

Kanter BR, Lykken CM, Polti I, Moser M-B, Moser EI (2025). Electrophysiological
recordings from entorhinal cortex and hippocampus in freely behaving adult male
rats. Data descriptor (data-descriptor_eebc19898ded.pdf, v1.0), EBRAINS.
<https://doi.org/10.25493/GTVB-FQR>

Data: Kanter B, Lykken C, Polti I, Moser M-B, Moser E (2025).
Electrophysiological recordings from entorhinal cortex and hippocampus in
freely behaving adult male rats (v1.0) [Data set]. EBRAINS.
<https://doi.org/10.25493/GTVB-FQR>
