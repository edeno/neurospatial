# Gardner et al. (2022): rat grid modules in a 1.5 m open field

**The question.** How do grid cells of one module share a firing pattern,
and how do modules differ in spacing? This dataset holds hundreds of
simultaneously recorded grid cells per rat, already sorted into the authors'
grid modules, with head direction tracked alongside position. It suits grid
scores and spacing per module, conjunctive grid × direction tuning and
population decoding.

| | |
| --- | --- |
| Dataset id | `gardner_2022` |
| Animal and region | Long Evans rats Q, R and S; medial entorhinal cortex-parasubiculum, layers II and III |
| Apparatus | 1.5 x 1.5 m open-field box with 50 cm walls |
| Archive | Figshare 16764508, version 6 (CC0); one 320 MB zip, of which a session reads two members (69 to 169 MB) |
| Sessions | 4 recordings, each loaded as its open-field session. The default `rat_r_day1` (483 units in modules 1 to 3) is checked by hand |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'gardner_2022'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]]
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("gardner_2022")
rates = compute_spatial_rates(
    rec.env,
    rec.spike_times,
    rec.times,
    rec.positions,
    min_speed=rec.params["min_speed"],
    epochs=rec.params["analysis_epochs"],
)
module_1 = rec.units["module"] == 1
```

## What was recorded

Neuropixels units in layers II and III of the medial entorhinal
cortex-parasubiculum region of three rats, four recordings in all (rat R on
two days). In each recording the rat stayed connected while it foraged in the
open-field box and ran a wagon-wheel track or slept in a 40 x 40 cm box, "each
in a different arena within the same room" (the paper's Methods). Head position
and head direction come from OptiTrack motion capture of five markers on the
implant, projected onto the horizontal plane, every 0.01 s (100 Hz).

The archive keeps only the units the paper assigned to grid modules, pure and
conjunctive grid cells together: 163 for rat Q (modules 1 and 2), 483 for rat R
day 1 and 544 for rat R day 2 (modules 1 to 3), and 140 for rat S (module 1),
the paper's "from 140 to 544 grid cells per session". `rec.units["module"]`
and `rec.params["module"]` give each unit's module; units are named
`mod<module>_<index>` after the file's keys.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| spike trains: read from the pickled `spikes_mod<m>` arrays with a restricted unpickler | stated | The pickles import only NumPy's array reconstruction (`_reconstruct`, `ndarray`, `dtype`), listed from their opcodes before any runs. `np.load(..., allow_pickle=True)` is never used. |
| samples and spikes of the open-field session only | stated | The file's `<rat>_sessions.txt` gives `open_field_1`'s start and end. The other sessions are on other apparatus and are named under "Watch out". |
| positions: x, y in m, times 100 to cm | inferred | No source read states the unit. The open-field samples span 1.43 to 1.48 m (rat R day 2: 1.56 m) against the stated 1.5 m box. |
| headings: `azimuth` in [0, 2π) wrapped to [-π, π], allocentric in the positions' frame | inferred | No source read states its zero or sense. Minus the direction of movement while running, its circular mean is within 5 degrees of 0 in every recording. |
| `epochs["open_field"]` and `params["analysis_epochs"]` | stated | The session's `valid_times` in the sessions text (rat R day 1), else its start and end. |
| environment: the stated 150 x 150 cm square on the samples' minimum rotated rectangle, 3 cm bins | inferred | The files state no origin or rotation. The `Conversion` gives each fitted side against 150 cm. |

The sampling rate (100 Hz) is **inferred** from the files' timestamps; no text
read states it. The units check refuses a session whose largest span is not
within 20% of the stated 150 cm side.

## Watch out

- **Pickled spike trains.** The spike arrays can only be read with
  `allow_pickle=True` in NumPy, which lets a file run code. The loader reads
  them with its own unpickler instead, which refuses anything but NumPy arrays.
  Do the same if you read the files yourself.
- **One session per recording.** Each file also holds wagon-wheel or
  sleep-box sessions; the loader keeps only the open-field session, and the
  summary lists the others with their times.
- **Times left out by `valid_times`.** For rat R day 1 the sessions text marks
  14778 to 14890 s of the open field as not valid, without saying why (the
  Methods say the task was paused to untwist the tether when needed). Those
  112 s are kept, outside `params["analysis_epochs"]`.
- **Module 1 of rat R day 1** is the paper's R1, which its clustering found as
  two clusters (R1a and R1b) of similar spacing and orientation; the file does
  not say which unit is in which.
- **Only grid-module units.** The archive holds no other units, so measures
  over all recorded MEC units cannot be computed.
- **Repeated spike times.** Some units hold a spike at exactly the time of the
  spike before it (1 in rat R day 1's open field, 7 in rat Q's). They are
  kept.

## Analysis choices

`rec.params`, from the Methods' "Spatial position and direction tuning":
`bin_size` 3 cm ("a square grid of 3 x 3-cm bins"); `min_speed` 2.5 cm/s
("only time epochs in which the rat was moving at a speed above 2.5 cm s-1
were used"); `analysis_epochs`, the open field's valid times; `dt` 0.25 s and
`train_epochs` / `test_epochs`, alternating 60 s blocks inside the analysis
epochs, for held-out decoding; `module`, each unit's grid module.

## Worked result

**Status: not possible yet.**

The paper's result is the toroidal topology of each module's population
activity, found by persistent cohomology, which neurospatial does not compute.
The one grid score the text states (mean 0.677 ± 0.017 over 618 cells,
Extended Data Fig. 2f) is of open-field maps masked to the bins visited on the
wagon-wheel track, pooled over all four recordings; that masking and the
paper's rate-map interpolation are not neurospatial steps, and two of the
recordings have no wagon-wheel session in their file.

## Cite

Gardner RJ, Hermansen E, Pachitariu M, Burak Y, Baas NA, Dunn BA, Moser M-B,
Moser EI (2022). Toroidal topology of population activity in grid cells.
Nature 602, 123. <https://doi.org/10.1038/s41586-021-04268-7>

Data: Gardner, Richard; Hermansen, Erik (2021). Toroidal topology of
population activity in grid cells. figshare. Dataset.
<https://doi.org/10.6084/m9.figshare.16764508.v6>
