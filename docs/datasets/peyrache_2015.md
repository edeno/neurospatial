# Peyrache et al. (2015): mouse head-direction cells in the anterior thalamus

**The question.** Do neurons in the anterior thalamus fire as a mouse faces
particular directions, and does that organisation persist in sleep? This dataset
is one session with head direction during wake and the wake, sleep, slow-wave
and REM epochs. It suits head-direction tuning, and decoding of heading from
spikes. It has no x/y positions.

| | |
| --- | --- |
| Dataset id | `peyrache_2015` |
| Animal and region | mouse, anterior thalamus, mainly the antero-dorsal nucleus (ADn) |
| Apparatus | 53 x 46 cm open arena, between two sleep sessions |
| Archive | OSF `jb2gd`, 36.6 MB, in the "Pynapple-files" project (G. Viejo); the file states no license |
| Sessions | 1: `Mouse32-140822`, checked by hand |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

catalog().query("dataset == 'peyrache_2015'")[["session", "n_units", "size_bytes", "verified"]]
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial.datasets import load

rec = load("peyrache_2015")
adn = rec.units["brain_region"] == "adn"      # the 31 units pynapple's tutorial keeps
print(rec.env, rec.n_units, adn.sum())        # None 49 31
```

## What was recorded

Units in the anterior thalamus of mice, mainly ADn, while they foraged in a
53 x 46 cm arena for 30 to 45 minutes between two sleep sessions of about 2
hours (the paper's Methods). Head direction is the orientation of a blue and a
red LED on the head, filmed at 30 frames/s and resampled at 39 Hz by the
acquisition system. Spikes were sorted with KlustaKwik and adjusted by hand.

The default session, `Mouse32-140822`, is the file that pynapple's and NeMoS's
head-direction tutorials download. It holds 49 units (31 with location `adn`
and 18 `thalamus`), the wake and sleep epochs, slow-wave and REM sleep
intervals, and head direction `ry` during the wake epoch only. It has no x/y
position, so the recording has no positions and `rec.env` is `None`.

DANDI 000056 (CC-BY-4.0) holds the dataset's other sessions, but not
`Mouse32-140822`, in a layout this loader does not read: raw blue and red LED
positions in camera pixels, with no head-direction series, no wake or sleep
epochs and no unit locations. They are not listed.

Sources read: the paper (Peyrache, Lacroix, Petersen & Buzsáki 2015, the author
manuscript on PubMed Central, PMC4376557), pynapple's head-direction tutorial,
the OSF project and file metadata, DANDI 000056's metadata, and the files.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| headings: `ry` (radians, 0 to 2 pi) wrapped to [-pi, pi] with `neurospatial.stats.wrap_angle` | stated | The file's series. |
| headings: allocentric, 0 = East and counter-clockwise, the package convention; not rotated or mirrored | inferred | The file states no reference frame, and the paper says only that heading is the orientation of two LEDs. |
| epochs: `wake` and `sleep` from the epochs table's tags; `sws` and `rem` from their own tables, times as stored, in seconds | stated | The file's tables. |

## Watch out

- **Two kinds of units.** 18 of the 49 units have location `thalamus`, not
  `adn`. The pynapple tutorial keeps the 31 `adn` units (unit ids 4 to 34) for
  head-direction analyses, so select `units["brain_region"] == "adn"` to match
  it.
- **Only copy.** This session is not in DANDI 000056 (which holds
  Mouse32-140820), so the OSF file is its only public copy and cannot be
  cross-checked against a second archive.
- **The heading frame is assumed, not stated.** 0 = East and counter-clockwise,
  the package convention. Tuning-curve shapes, widths and decoding error do not
  depend on it; absolute preferred directions, and comparing them with another
  session's or a cue's, do. With no positions, heading cannot be compared with
  the direction of movement.
- **Tracking gaps.** Head direction exists only during the wake epoch, and the
  summary counts the gaps longer than a set length, with their total and
  longest. Spikes in a gap have no heading; nothing is filled in.

## Analysis choices

`rec.params`: `analysis_epochs`, the wake epoch, the only time with tracked head
direction; `dt` 0.1 s, the decoding bin of pynapple's head-direction tutorial.

## Worked result

**Status: reproduced. Level A (a number compared with the paper's).**

Finding: the share of ADn units that are head-direction cells.

The paper: "Neurons were classified as HD cells for concentration parameter
exceeding 1, peak firing rates greater than 1Hz and probability of non-uniform
distribution smaller than 0.001." ... "In the thalamus, when considering only
cell ensembles recorded on silicon probe shanks where at least one HD cell was
detected, the average proportion of HD cells was 71%".

| | Value |
| --- | --- |
| Paper | 71% |
| Ours | 20 of the 31 `adn` units (64.5%) |
| Criterion (fixed before the run) | the absolute difference from 0.71 is at most 0.163. The 71% is an average over sessions with no stated spread, so the tolerance is two binomial standard deviations for one session of 31 units, 2 * sqrt(0.71 * 0.29 / 31) |

How it was computed: the 31 `adn` units in the wake epoch,
`compute_directional_rates` with 1 degree bins and a 6 degree Gaussian (the
paper's); concentration above 1, as mean vector length above I1(1) / I0(1) =
0.446; peak rate above 1 Hz; `stats.rayleigh_test` on the headings at the
unit's spikes, each the nearest tracked sample's, p < 0.001. LED dropouts
("Times when one of the LEDs was not detected were excluded from the
analysis") are the 64 sample intervals longer than 1.5 sampling periods (0.0384
s; the others are 0.025 to 0.026 s): left out of the tuning curves
(`max_gap`) and their spikes out of the Rayleigh test.

Not matched to the paper:

- the denominator: the file's `adn` units, not the units of shanks with at
  least one HD cell (the recording holds no shank);
- the von Mises fit: the paper does not say how it fitted; this is the
  maximum-likelihood fit to the tuning curve read as a distribution of angles,
  whose concentration kappa has A1(kappa) equal to the curve's mean vector
  length;
- the peak rate: the smoothed curve's, not the fitted von Mises's;
- the Rayleigh test: the paper does not say on what; this is on the headings at
  the spikes, not on the occupancy-normalized field.

Source: Peyrache et al. 2015, Methods, Data analysis and statistics, "Head
direction cells" (author manuscript, PMC4376557).

## Cite

Peyrache A, Lacroix MM, Petersen PC, Buzsáki G (2015). Internally organized
mechanisms of the head direction sense. Nature Neuroscience 18, 569-575.
<https://doi.org/10.1038/nn.3968>

Data: Viejo, Guillaume (2023) Pynapple-files: NWB files to be used for the
pynapple and GLM package documentation. OSF. <https://osf.io/tkzr7/>
