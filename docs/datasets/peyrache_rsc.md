# Peyrache lab: mouse retrosplenial cortex with head direction and position

**The question.** How do retrosplenial cortex (RSC) neurons encode where a mouse
is and which way its head points, and how do they behave in sleep? This dataset
is 14 sessions from three mice that alternate sleep in the home cage with
exploration of a square or a circular arena, the head tracked in 3-D by motion
capture. It is the first dataset in the catalog with both head direction and
x/y position, so it suits head-direction tuning, conjunctive coding, and
comparing heading with the direction of travel.

| | |
| --- | --- |
| Dataset id | `peyrache_rsc` |
| Animal and region | mouse (3 animals), retrosplenial cortex |
| Apparatus | a square arena and a 70 cm circular arena, between sleep sessions in the home cage |
| Archive | DANDI 000987, version 0.261003.2342 (CC-BY-4.0); NWB files of 20 to 156 MB |
| Sessions | 14. One is checked by hand: the default `sub-A7803_ses-20240509T213129` (37 units, sleep then the square) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'peyrache_rsc'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load
from neurospatial.encoding import compute_directional_rates

rec = load("peyrache_rsc")
square = rec.epochs["square"]
rates = compute_spatial_rates(
    rec.env, rec.spike_times, rec.times, rec.positions, epochs=square
)
tuning = compute_directional_rates(
    rec.spike_times, rec.times, rec.headings, epochs=square
)
```

## What was recorded

Units in RSC of three mice (A7801, A7802, A7803), on silicon probes. Each session
starts with sleep in the home cage, then one or more arena epochs separated by
sleep. The file's `session_description` names the arenas in order, for example
"Order of epochs – sleep, square arena, circular arena, sleep, square arena,
circular arena, sleep"; the loader names each tracked epoch after its arena
(`square`, `circle`, `partial_circle`). The study's bioRxiv preprint
(Mehrotra, Levenstein et al., 10.1101/2023.09.12.557442, v1, Methods) says that
"For the RSC dataset, animals explored a circular arena (diameter 70 cm) with a
white cue card" "in addition to the square arena", and that the head was
tracked "in 3D using 7 infrared cameras (Flex 13, Optitrack)". Each file also
holds slow-wave sleep and UP and DOWN states.

DANDI's description says the files were used for Figure 5F of "Hyperpolarization-
activated currents drive neuronal activation sequences in sleep" (Current
Biology 2024). That paper was not read; every number on this page from a paper
is the preprint's.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: cm = (`Position/x`, -`Position/z`) x 100; `Position/y` is vertical and left out | inferred | The three series have a blank unit and reference frame. y spans a few cm (the head's height), x and z 50 to 80 cm. Read as metres, the circular epochs span 64 to 70 cm, the preprint's 70 cm circle. Motion capture with y up is right-handed, so seen from above x points East and -z North. The heading's movement test cannot check this: mirroring the positions and the heading together leaves heading minus movement unchanged. |
| headings: `CompassDirection/ry` wrapped from [0, 2π) to [-π, π] | inferred | ry is the rotation about the vertical axis. Negated, it spreads more around the direction of movement (87 against 49 degrees on the default session), so it turns the same way as the (x, -z) frame. Its zero is not verified (see below). |
| epochs: `sleep`, `wake` (the epochs table's tags), `sws`, `up`, `down`, `tracking` (`position_time_support`) | stated | The file's interval tables. |
| epochs: `square`, `circle`, `partial_circle`, the tracked epochs in the order the `session_description` lists the arenas | stated | The file's `session_description`; no arena epoch is named when the number of arenas and tracked epochs differ. |
| environment: built from the samples (`Environment.open_field`, 2.5 cm bins) | inferred | The preprint gives the circle's diameter, but not the side of the square ("a plastic platform with removable walls (width: 80 cm, height: 50 cm)"); the square epochs span 50 to 56 cm. |

The sampling rate is **inferred**. The preprint states "100 Hz (120 Hz for the
RSC dataset)", and 11 of the 14 files sample at 119.76 Hz, but the three A7803
sessions that visit only the square (A7803-210722, -210723 and -210726, the
default) sample at 100 Hz. Those three load with a watch-out naming the
difference. The position span is inferred from the files (largest axis 50 to 83
cm); a session outside 40 to 100 cm loads with a watch-out.

## Watch out

- **The heading frame is not verified by movement.** The loader compares heading
  with the direction of movement above 10 cm/s, at 10 Hz, in each tracked epoch.
  On the default session the circular mean of heading minus movement is +0.51
  rad and the spread 49 degrees, outside the rule (|mean| at most 0.2 rad, a
  spread under 45 degrees). In sessions with several arena epochs the mean
  changes from epoch to epoch, by more than 2 rad in some, so the rigid body's
  zero moved. The headings are kept as stored. Tuning-curve widths and mean
  vector lengths do not depend on the frame; preferred directions, and
  comparisons across epochs, do. The summary prints the numbers for each
  session.
- **One environment for two arenas.** Most sessions visit a square and a circle
  (or a partial and a full circle), tracked in one coordinate frame, so
  `rec.env` covers both. For one arena's maps build its environment from its
  epoch's samples and pass the epoch:

  <!-- docs-test: skip needs a download -->
  ```python
  import numpy as np
  from neurospatial import Environment, compute_spatial_rates
  from neurospatial.datasets import load

  rec = load("peyrache_rsc", "sub-A7803_ses-20240509T221227")
  circle = rec.epochs["circle"]
  t = rec.times[:, None]
  inside = np.any((t >= circle[:, 0]) & (t < circle[:, 1]), axis=1)
  env = Environment.open_field(rec.positions[inside], bin_size=2.5, units="cm")
  rates = compute_spatial_rates(
      env, rec.spike_times, rec.times, rec.positions, epochs=circle
  )
  ```
- **The sampling rate differs from the preprint's** in the three square-only
  A7803 sessions, the default among them (100 Hz, not 120 Hz).
- **Held tracking.** The default session has 22 runs of 11 or more identical
  position samples (0.43% of the frames, the longest 1.0 s): the motion-capture
  system repeating its last fix, or a still mouse. They are kept as stored.
- **The square's size is not stated.** The default session's square spans
  52.7 x 53.2 cm; the preprint gives the wall panels' width (80 cm), not the
  side of the square they formed, so the environment is not declared.
- **The file's date.** Every file's `session_start_time` is 2024-05-09, minutes
  apart, most likely when the files were written; the identifier
  (`A7803-210726`) reads as the recording date, 2021-07-26. Times are seconds
  from the file's start.

## Analysis choices

`rec.params`: `bin_size` 2.5 cm; `analysis_epochs` the tracked epochs
(`position_time_support`), so sleep is left out of the rate maps; `min_speed`
`None` (no source read states a running threshold); `dt` 0.25 s and
alternating 60 s `train_epochs` and `test_epochs`.

## Worked result

**Status: not possible yet.**

DANDI says the files were used for Figure 5F of the Current Biology paper,
which was not read. The preprint, which was, analyses the RSC units' sleep
dynamics pooled over sessions and gives no spatial or head-direction number for
one session to check.

## Cite

Mehrotra D, Levenstein D, Duszkiewicz AJ, Carrasco SS, Booker SA, Kwiatkowska A,
Peyrache A (2024). Hyperpolarization-activated currents drive neuronal
activation sequences in sleep. Current Biology 34, 3043–3054.e8.
<https://doi.org/10.1016/j.cub.2024.05.048>

Data: Mehrotra, Dhruv; Skromne Carrasco, Sofia; Peyrache, Adrien (2026) Freely
moving mouse retrosplenial cortex recordings (Version 0.261003.2342) [Data set].
DANDI Archive. <https://doi.org/10.48324/dandi.000987/0.261003.2342>
