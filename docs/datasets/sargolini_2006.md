# Sargolini et al. (2006): rat medial entorhinal cortex in a 1 m box

**The question.** Do cells in medial entorhinal cortex (MEC) fire on a grid
across an open field, and do some also carry head direction? This dataset is
118 sessions of rats running in a 1 x 1 m enclosure. It suits spatial rate
maps and the grid-cell metrics.

| | |
| --- | --- |
| Dataset id | `sargolini_2006` |
| Animal and region | Long Evans rats, medial entorhinal cortex |
| Apparatus | 1 m x 1 m open-field enclosure; sessions are normally 10 minutes (the authors' readme) |
| Archive | DANDI 000582, version 0.251111.2151 (CC-BY-4.0); NWB files of about 4 to 19 MB |
| Sessions | 118. Two are checked by hand: the default `sub-10073_ses-17010302` (8 units) and `sub-10884_ses-03080402` (6 units) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'sargolini_2006'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("sargolini_2006")
rates = compute_spatial_rates(rec.env, rec.spike_times, rec.times, rec.positions)
```

## What was recorded

Units in MEC of Long Evans rats running in a 1 x 1 m enclosure. DANDI's
description says the authors recorded from each principal cell layer of MEC;
the files' `experiment_description` says the sample includes conjunctive and
head-direction cells from layers III and V, and each unit's histology names its
layer (MEC LII, LIII, LV or LVI in the 12 sessions opened). Position is one
tracking LED (`SpatialSeriesLED1`), sampled every 0.02 s (50 Hz); some files
also hold the second LED (`SpatialSeriesLED2`).

The default session, `sub-10073_ses-17010302`, is the one pynapple's DANDI
tutorial streams: 8 layer II units. `sub-10884_ses-03080402`, used by
pynapple's null-distribution tutorial, is also checked by hand.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: cm = first LED's series in its stored unit times the file's conversion, times 100 for metres | stated | The file stores cm with `unit="meters"` and `conversion=0.01`; `read_position` returns metres, multiplied by 100 to give cm. |
| environment: the stated 100 x 100 cm square centred at the midrange of the tracked samples, sides along the file's x and y, 2.5 cm bins | inferred | The file states no origin or rotation. The `Conversion` gives how far each side of the samples' range falls short of or past 100 cm. |
| headings | none | `rec.headings` is `None`. Neither the file nor the authors' readme says which of two LEDs is at the front, so head direction is not computed, even where both are stored. |

The sampling rate (50 Hz) is **inferred** from the files' timestamps, 0.02 s
steps in 12 of the 118 sessions; no text read states it. A session at another
rate loads with a watch-out naming the evidence. The stated position span is
the 1 m side, and the units check refuses a session whose largest span
disagrees with it by more than 20%.

## Watch out

- **Sessions that fail the units check.** Of the 12 sessions opened, two are
  1200 s long and span about 1.5 m and 2 m (`sub-11265_ses-16030601`,
  `sub-11207_ses-24060501`), though their session_description states the
  1 x 1 m enclosure. No source read states the box they were recorded in, and
  the units check refuses them. `accept_failed_checks=("units",)` loads one
  anyway.
- **No headings and no head-direction cells.** A file with only the first LED
  says so in its summary; the paper's head-direction and conjunctive cells
  cannot be found from it. A file with the second LED stores it but does not
  say which LED is at the front.
- **Positions already fitted to the box.** In the 10-minute sessions opened so
  far, the positions are centred on their own midrange and the longer axis
  spans exactly 100 cm, so the coordinates were centred and scaled to the box
  before publication. They do not show where the walls were, and the span
  cannot reveal a scale error. The summary says when a session shows this.
- **Layer.** For `sub-10073_ses-17010302` the file's `experiment_description`
  says the sample is from layers III and V, but this session's units table
  gives every unit's histology as MEC LII; `units["brain_region"]` follows the
  units table.
- **A NaN sample.** In `sub-10884_ses-03080402` the first of the 30000 position
  samples, at 0 s, is NaN (not tracked); it is kept as NaN.

## Analysis choices

`rec.params`: `bin_size` 2.5 cm (40 bins across the box, as in the
spatial-information table pynapple's tutorial prints for
`sub-10884_ses-03080402`); `dt` 0.25 s, the decoding bin; `min_speed` `None`
(no speed filter, as in that table); `train_epochs` and `test_epochs`,
alternating 60 s blocks, so decoding is always tested on held-out time.

## Worked result

**Status: not possible yet.**

The candidate result is the grid scores of the session's grid cells. It waits
because the paper (Sargolini et al. 2006, Science 312, 758-762) could not be
read: the local library holds its metadata only, science.org refuses access
(HTTP 403), and Europe PMC lists it without full text. Only what was read is
cited, so no result is claimed.

## Cite

Sargolini F, Fyhn M, Hafting T, McNaughton BL, Witter MP, Moser M-B, Moser EI
(2006). Conjunctive representation of position, direction, and velocity in
entorhinal cortex. Science 312, 758. <https://doi.org/10.1126/science.1125572>

Data: Sargolini, Francesca; Fyhn, Marianne; Hafting, Torkel; McNaughton, Bruce
L.; Witter, Menno P.; Moser, May-Britt; Moser, Edvard I.; Waade, Haagen; Ball,
Simon (2025) Conjunctive Representation of Position, Direction, and Velocity in
Entorhinal Cortex (Version 0.251111.2151) [Data set]. DANDI Archive.
<https://doi.org/10.48324/dandi.000582/0.251111.2151>
