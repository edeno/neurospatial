# Tirole et al. (2022): rat CA1 replay of two novel linear tracks

**The question.** A place cell often fires faster on one track than on another.
Does that difference in rate come back when the two tracks are replayed? These
are tetrode recordings from dorsal CA1 of five rats, each session a rest, runs
on two novel 2 m linear tracks, and a long rest. The dataset suits replay of
two environments in one session, decoded jointly, and the authors' own replay
pipeline can be compared event by event.

| | |
| --- | --- |
| Dataset id | `tirole_2022` |
| Animal and region | Five male Lister-Hooded rats; dorsal CA1 (two rats also had 8 of 24 tetrodes in left V1, which the files do not mark) |
| Apparatus | Two novel 2 m linear tracks, bent into a new shape each session, in one room; a 20 cm rest pot |
| Archive | Zenodo 7114733 (CC0), the mirror of Dryad 10.5061/dryad.ksn02v76h: per session a spike file and a position file (20 to 45 MB together), and a 1.2 to 1.7 GB LFP file on request |
| Sessions | 9 of Table 1's 10, under their Table 1 names. One is checked by hand: the default `rat3_session2` (142 units) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'tirole_2022'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]]
```

<!-- docs-test: skip needs a download -->
```python
import numpy as np

from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("tirole_2022")
track = rec.params["tracks"]["track_1"]  # 200 cm in 10 cm bins
along_track = np.column_stack([rec.linearized, np.zeros_like(rec.linearized)])
rates = compute_spatial_rates(
    track,
    rec.spike_times,
    rec.times,
    along_track,  # NaN off the tracks
    epochs=rec.epochs["track_1"],
    min_speed=rec.params["min_speed"],
    method="binned",
)

samples, times = rec.lfp.fetch(rec.lfp.ripple_channels)  # the authors' ripple channel
```

## What was recorded

A session started with about an hour in the rest pot (`rec.epochs["pre"]`).
The rat then ran back and forth on a novel track for about 15 minutes
(`"track_1"`), rested about 10 minutes (`"between_tracks"`; Rat 2 session 2
went straight on), ran a second novel track (`"track_2"`), and rested about two
hours (`"post"`). Chocolate soy milk at both ends of a track kept it running. In
the paper's second protocol (the table below) the rat first ran a third track,
which the paper removed and the deposit does not hold; `"pre"` then spans that
run, as Table 1's PRE does.
Rat 2 session 1 ran both tracks again after its POST rest
(`"track_1_reexposure"`, `"track_2_reexposure"`).

`rec.epochs["rest_pot"]` holds the samples the file marks in the rest pot, as
stored: runs broken wherever the tracking lost the rat. The file does not mark
sleep: the authors scored it with thresholds set by hand that are not in the
deposit.

| Session | Spike and position files | LFP file | Protocol |
| --- | --- | --- | --- |
| `rat1_session1` | `RAT1_SESS2` | `RAT1_SESS1` | third track first |
| `rat2_session1` | `RAT2_SESS1` | `RAT2_SESS1` | two tracks, re-exposure after POST |
| `rat2_session2` | `RAT2_SESS2` | `RAT2_SESS2` | two tracks, no rest between |
| `rat3_session1` | `RAT3_SESS1` | `RAT3_SESS1` | two tracks |
| `rat3_session2` | `RAT3_SESS2` | `RAT3_SESS2` | two tracks |
| `rat4_session2` | `RAT4_SESS2` | `RAT4_SESS2` | third track first |
| `rat4_session3` | `RAT1_SESS1` | `RAT4_SESS3` | two tracks, 30 min on the second |
| `rat5_session1` | `RAT5_SESS1` | `RAT5_SESS1` | third track first |
| `rat5_session2` | `RAT5_SESS2` | `RAT5_SESS2` | third track first |

Sources read: the paper (eLife 11:e79031, Methods and Table 1), the deposit's
`README.txt`, the files, and the authors' code
(github.com/bendor-lab/Elife_Tirole_Huelin_Gorriz_2022 at 44ecf42, CC0).

## The deposit's labels are crossed

The deposit names its files `RATn_SESSm`, but two pairs are mixed up. The loader
pairs each spike and position file with the LFP file whose clock spans the same
seconds, and Table 1 confirms each pair: the authors' lap counter, run on the
position file, gives Table 1's laps on both tracks (it counts one-way runs, half
of which are Table 1's laps), and PRE and POST last Table 1's minutes (the test
`test_rest_durations_are_table_1s` checks all nine).

- Rat 1 session 1 is the spike and position files `RAT1_SESS2` (the deposit's
  `RAT4_SESS1` files are byte-identical copies) with the LFP `RAT1_SESS1`.
- Rat 4 session 3 is the spike and position files `RAT1_SESS1` with the LFP
  `RAT4_SESS3`.
- **Rat 4 session 1 is not offered.** Only its LFP (`RAT4_SESS1_CSC`) is its
  own; its spike and position files in the deposit are Rat 1 session 1's.
  Table 1 gives it 52 stable cells, the most of any session.

## Units and conversions

- **Positions** are the camera's pixels on the regular 25 Hz grid
  `position.t`, times the file's `position.cm_per_pixel`, which the README calls
  the "average cm/pixel value across tracks". The axes are the camera's: `y`
  runs down the image, so a plot with `y` up is mirrored.
- **Linearized position** is the authors' own, 0 to 200 cm along each track, NaN
  off the tracks. The authors gave each track its own scale, 200 cm over the
  line they drew along it, so it differs from the positions' average scale by
  up to 9% (a watch-out lists each track's).
- **Spike times** are in seconds on the tracking clock. `rec.units` keeps the
  authors' unit ids (1 to N), so they match the authors' code, with the tetrode
  and cluster from `id_conversion`.
- **LFP** is four channels at 1 kHz, the channels the authors chose for ripple,
  theta, delta and the theta reference, stored without filtering. The README
  says "in Volts ?", but the values run to the thousands, so the unit is
  arbitrary. The files of every session but Rat 2 session 1 are MATLAB v7.3,
  read with `h5py`.

The environment is declared: both tracks of the session as one graph, each the
line the authors drew along it to linearize it, simplified to its corners, with
2 cm bins. The rest pot lies off the tracks, so most of a session's samples fall
in no bin. `rec.params["tracks"]` holds each track as the 200 cm line of 10 cm
bins the authors decoded on, for positions `(rec.linearized, 0)`.

## Watch out

- The labels are crossed in two sessions (above), and Rat 4 session 1 cannot
  be loaded.
- The deposit holds sorted single units only. The paper's candidate events came
  from multi-unit activity that included the clusters it classed as multi-unit,
  which are not deposited, so a multi-unit rate from these files has fewer
  spikes than the paper's.
- No waveforms are deposited, so the paper's pyramidal-cell filter (half width
  over 500 µs) cannot be applied. The authors' code falls back to the rate rule
  alone (below 5 Hz) without them.
- Each track's own scale differs from the average scale of the positions.
- Rat 5 session 2's PRE and POST are Table 1's the other way round.
- Rat 2 session 1's LFP file ends at 100,629.6 s, before its re-exposure.

## Analysis choices

- `bin_size` 2 cm, the paper's rate maps for display; `decoding_bin_size` 10 cm,
  its rate maps for decoding.
- `min_speed` 5 cm/s and `max_speed` 50 cm/s: the speeds of the rate maps in the
  authors' code. The paper's text says 4 to 50 cm/s.
- `dt` 0.02 s and `run_dt` 0.25 s, the paper's decoding bins in replay and in
  running.

## Worked result

Not yet. The paper's Figure 3A regression (B = 0.05, F(1,268) = 186, R² = 0.41)
needs the replay pipeline's judged events and place fields.

## Cite

Tirole M, Huelin Gorriz M, Takigawa M, Kukovska L, Bendor D (2022).
Experience-driven rate modulation is reinstated during hippocampal replay.
eLife 11:e79031. https://doi.org/10.7554/eLife.79031

Data: Takigawa M, Kukovska L, Tirole M, Huelin M, Bendor D (2023).
Experience-driven rate modulation is reinstated during hippocampal replay.
Dryad, mirrored on Zenodo. https://doi.org/10.5061/dryad.ksn02v76h
