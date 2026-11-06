# Comrie et al. (2026): rat hippocampus in a spatial bandit maze

**The question.** While a rat runs a maze with three Y-shaped patches, does the
hippocampal population represent the place where the rat is? This dataset is
one run epoch of one rat, with 203 sorted hippocampal units and a track graph
the authors drew, so it suits decoding on a track with branches.

| | |
| --- | --- |
| Dataset id | `comrie_2026` |
| Animal and region | rat j16, hippocampus (the record's word; the preprint's tetrodes are in dorsal CA1) |
| Apparatus | spatial bandit maze: three Y-shaped patches of 6.5 cm wide, 53 cm hallways meeting at 120 degrees, six reward ports |
| Archive | Zenodo 23020757, version 1.0 (CC-BY-4.0), one `.npz`, 75.1 MB |
| Sessions | 1: `j1620210710_02_r1_figure04_inputs`, checked by hand |
| Needs | no extra: the file is a NumPy `.npz`, read without the `datasets` extra |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

catalog().query("dataset == 'comrie_2026'")[["session", "n_units", "size_bytes", "verified"]]
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial.datasets import load

rec = load("comrie_2026")
print(rec)
```

## What was recorded

Rat j16 on 2021-07-10, run epoch `02_r1` of the spatial-bandit task, with 203
spike-sorted hippocampal units. The head position is the centroid of a front
and a back LED on the implant, filmed at 30 frames/s and, in the authors'
pipeline, converted to cm, smoothed, interpolated and upsampled to 500 Hz. The
file also holds the authors' linearized position and the track graph they
linearized with: 10 nodes and 9 edges, with nodes at the reward ports and the
hallway intersections.

The file was published with Zeng, Comrie, Frank, Eden and Denovellis, *Local
goodness-of-fit measures for neural decoding* (its Figure 4 input). The raw
recording is DANDI 001942; the spike sorting is only in this file. The record
states it has no pickled objects, and the loader reads it with
`np.load(..., allow_pickle=False)`.

Sources read: the Zenodo record's description, the file, and the bioRxiv
preprint of Comrie et al. (doi:10.1101/2024.09.23.613567, version 1, read on
PubMed Central, PMC11463554). The published Neuron article was not available
to read, so the maze and tracking facts here are the preprint's.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: `head_position_x` and `head_position_y` as stored, in cm | inferred | Neither the record nor the file states the unit. The graph's edges, in the positions' unit, are 52 to 57 long against the 53 cm hallways the preprint states, and the preprint says positions were converted to cm. The file's `head_speed` matches the speed of the stored positions in shape (correlation 0.93) and in scale (medians 6.39 and 8.04 cm/s, a factor of 1.26), so speed and position share the unit. The loader refuses the file if the correlation falls below 0.8 or the ratio of the medians moves more than a factor of 2 from the 1.26 measured on this file: a unit error moves it by a factor of 2.54 (inch), 10 (mm) or 100 (m), and a correlation alone cannot see it. |
| spike trains: `spike_times` cut at `spike_offsets` into 203 units | stated | The Zenodo record: "the spike times of 203 units, 870,018 spikes (spike_times, split into units by spike_offsets)". 21 units have no spikes in this epoch; they stay as empty trains. |
| environment: the authors' track graph, every edge kept, in the file's `linear_edge_order` with `linear_edge_spacing` 15 cm, 2 cm bins | stated | The file's graph; the preprint's nodes at ports and intersections. Edge lengths are 51.7 to 57.4 cm against the stated 53 cm, and `env.to_linear` reproduces the file's `linear_position` exactly. |
| environment: `track_width`, a placement tolerance, not the 6.5 cm hallway width | inferred | Twice the 99th percentile distance of the head samples from the graph. By construction a known small share of samples falls in no bin. |
| times | none | Kept as stored: seconds on the Unix clock (1.6259e9 s is 2021-07-10), samples every 0.002 s. |
| headings | none | `rec.headings` is `None`; see "Watch out". |

## Watch out

- **The units are one epoch's sorting.** They are the authors' spike sorting of
  this one run epoch. The raw recording is DANDI 001942, which does not hold
  this sorting, so these units cannot be matched to another epoch's.
- **No headings.** The file's `head_orientation` (from the front and back
  LEDs), minus the direction of movement above 10 cm/s, has a circular mean of
  +13.1 degrees: the head's direction with a constant offset, from the LEDs'
  mounting or the rat's head posture, which the file cannot tell apart. It is
  kept in `rec.params["head_orientation"]` (radians); subtract the offset only
  if it is the mounting.
- **Many head samples lie beyond the hallways.** About 64% of head samples are
  farther than half the 6.5 cm hallway width from the track graph, most when
  the rat is stopped. This is tracking spread, not misfit: the head swings past
  the hallways' low walls and a raised head projects outward, while each edge's
  samples are centred on it. The positions are kept as stored, and the
  summary gives the split by speed.
- **Smoothed positions.** Positions are the authors' smoothed, interpolated
  head positions, upsampled to 500 Hz from a 30 frames/s camera, about 16
  samples per video frame. Speeds and held-sample counts are those of smoothed,
  interpolated positions, not of raw tracking.

## Analysis choices

`rec.params`: `bin_size` 4 cm (a 2-D grid over the maze) and `graph_bin_size`
2 cm (along the track, the environment's bins); `min_speed` 10 cm/s, the
preprint's Methods threshold for locomotion ("valid times of high speed
(>10 cm/s)"); `dt` 0.25 s, the decoding bin; `train_epochs` and `test_epochs`,
alternating 60 s blocks; and `head_orientation`.

## Worked result

**Status: a test against chance, not a finding of the preprint. Level B.**

While the rat runs, the decoded position usually lies on the rat's own track
segment. This is a property of the data and the decoder, not something the
preprint reports as a result; the preprint's *Non-local representation* is
defined against it: "Non-local representation times were identified as times
in which the most likely decoded position (hereafter "decoded position") was
in a track segment distinct to the animal's actual position." (Methods,
"Non-local representation of Stay and Switch paths"). The preprint also says
that place cells' activity "typically signals the actual location of the
animal" (Introduction). The preprint states no share to compare with.

| | Value |
| --- | --- |
| Ours | 92.4% of held-out running decoding bins have the MAP bin on the rat's own segment |
| Paper | no share stated |
| Criterion (fixed before the run) | the share exceeds both a constant decoder that names the segment the rat runs on most (35.1% of those bins) and the 95th percentile of the same share after shifting every unit's spikes by one common circular offset (20 draws, seed 2026: 18.1% to 29.4%, 95th percentile 28.1%) |

How it was computed: a `BayesianDecoder` on `rec.env` (the authors' track
graph, 2 cm bins, default diffusion KDE), `dt` 0.25 s, fitted on
`params["train_epochs"]` with `min_speed` 10 cm/s and decoded on
`params["test_epochs"]`; a bin counts when the speed at its centre is above
10 cm/s; the segment is the graph edge (`source_edge_id`) of the MAP bin and
of the rat's bin.

Not matched to the preprint:

- the decoder: the preprint's is clusterless and state-space, in 2 ms bins,
  fitted and decoded on the same data; this one uses the file's sorted units,
  0.25 s bins and held-out time;
- the preprint's confidence criterion (the highest 50% of the posterior within
  50 cm) is not applied.

Source: Comrie et al., bioRxiv 2024.09.23.613567 v1 (PMC11463554),
Introduction and Methods; the published Neuron article was not available to
read.

## Cite

Comrie AE, Monroe EJ, Kahn AE, Denovellis EL, Joshi A, Guidera JA, Krausz TA,
Berke JD, Daw ND, Frank LM (2026). Hippocampal representations of alternative
possibilities are flexibly generated to meet cognitive demands. Neuron.
<https://doi.org/10.1016/j.neuron.2026.08.023>

Data: Comrie, Alison; Frank, Loren; Denovellis, Eric (2026) Figure 4 input
data for "Local goodness-of-fit measures for neural decoding" (rat j16,
2021-07-10, run epoch 02_r1) (Version 1.0) [Data set]. Zenodo.
<https://doi.org/10.5281/zenodo.23020757>

`rec.citation` holds both.
