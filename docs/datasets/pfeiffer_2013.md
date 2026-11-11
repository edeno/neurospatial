# Pfeiffer & Foster (2013): place-cell sequences in a 2 m open arena (rat)

**The question.** When a rat pauses in an open arena, do CA1 place cells
replay two-dimensional paths, and do those paths lead to the goal it
remembers? Pfeiffer & Foster (2013) decoded brief population events into
"trajectory events" and found they ran from the rat's location toward the
day's Home well. This page loads the one session of their data that is public,
with Pfeiffer's own candidate windows and his own per-event significance
labels, so you can check a replay method against the author's verdicts.

| | |
| --- | --- |
| Dataset id | `pfeiffer_2013` |
| Animal and region | Long-Evans rat 1, day 1 (inferred); dorsal CA1, both hemispheres |
| Apparatus | 2 x 2 m black arena with 30 cm walls and 36 reward wells |
| Source | `data/openfield.nel` (55.8 MB), a sample recording released with Maboudi et al. (2018), fetched from eLife's archived copy of their repository at a pinned commit |
| License | not stated (sample released with Maboudi et al. 2018; the repository's MIT license covers the code) |
| Sessions | 1 (`rat1_day1`, 212 units), checked by hand |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'pfeiffer_2013'")
sessions[["session", "n_units", "size_bytes", "verified", "license"]]
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("pfeiffer_2013")
excitatory = [
    train
    for train, label in zip(rec.spike_times, rec.units["label"])
    if label == "excitatory"
]
rates = compute_spatial_rates(
    rec.env,
    excitatory,
    rec.times,
    rec.positions,
    method="binned",
    bandwidth=rec.params["bandwidth"],  # the papers' 4 cm Gaussian
    min_speed=rec.params["min_speed"],  # 5 cm/s
)
windows = rec.events["ripples"]  # Pfeiffer's 322 candidate windows
labels = rec.params["significant"]  # his 110 significant ones, as a mask
```

## What was recorded

Tetrodes in dorsal CA1 of both hemispheres while the rat alternated two
phases: foraging for a reward at an unknown (Random) well, then returning to
that day's Home well (the paper's Methods). Position comes from two
head-mounted LEDs filmed at 60 Hz; the file holds one point per sample, 57.4
samples per second, with no head direction. The paper states 212 units for
rat 1 on day 1, and the file holds 212: 203 labelled excitatory and 9
inhibitory by the file's own id lists (`rec.units["label"]`).

`rec.events["ripples"]` holds Pfeiffer's 322 candidate windows and
`rec.events["significant_ripples"]` the 110 of them his file marks
significant; `rec.params["significant"]` is the same as a mask over the 322.
The file also holds Maboudi's analysis (a multi-unit rate, 614 `mua_epochs`,
binned events and HMM scores), which is not loaded. There is no LFP, no well
location, no Home well and no trial or reward time.

## Where the data come from, and its terms

Maboudi et al. (2018, eLife 7:e34467), whose authors include Pfeiffer and
Foster, write under Data availability: "Data for Figure 6 was previously
reported in Pfeiffer and Foster (2015, Science) and is available from Brad
Pfeiffer and David Foster on request. ... All analysis code and sample
recording epochs for Figures 1-7 are available on
<https://github.com/kemerelab/UncoveringTemporalStructureHippocampus> (copy
archived at
<https://github.com/elifesciences-publications/UncoveringTemporalStructureHippocampus>)."
The file was added on 2018-05-16 by co-author Etienne Ackermann (commit
962b3870, "added open field example session and notebook").

**No data license is stated.** The repository's only license is MIT
("Copyright (c) 2018 Kemere Lab"), a software license for the code. The
loader downloads the file from eLife's archived copy at commit f1b021d,
checks its SHA-256 (the same file as the kemerelab original), and never
redistributes it. The paper's other seven sessions are available from the
authors on request.

## Reading the file safely

The file is a gzip `dill` pickle of `nelpy` 0.1.2 objects. Unpickling it the
usual way would run whatever code it names. The loader lists the pickle's
imports from its opcodes first and refuses any outside a short allow-list;
`nelpy`'s classes become inert containers holding their pickled state, and
`dill`'s helpers resolve only to NumPy's array and scalar reconstructors.
Neither `nelpy` nor `dill` is imported. Do the same if you read the file
yourself.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| times, spike times, windows: the nelpy objects' times, in seconds | inferred | nelpy stores seconds; position samples are a median 17.4 ms apart (a 60 Hz camera), and spikes, positions and windows share one span, 2572.0 to 4759.0 s |
| positions: x and y as stored, in cm | inferred | No source read states the unit; the samples span 198.6 x 191.6 in the stated 2 x 2 m arena |
| unit labels: `ExcitatoryNeuronIDs` and `InhibitoryNeuronIDs`, read as the file's unit ids (1 to 212) | inferred | The two lists are disjoint and together hold every id once; Krause & Drugowitsch (2022)'s code subtracts 1 from the MATLAB lists for its 0-based arrays |
| significant windows: `BradSignificantRippleIDs` as 0-based indices into the 322 | inferred | Krause & Drugowitsch's code loads MATLAB's `SignificantRipples - 1`; only offset 0 lines up with Maboudi's HMM significance per event |
| environment: the stated 200 x 200 cm arena on the samples' minimum rotated rectangle, 2 cm bins | inferred | The file states no origin or rotation; the fit is centred at (188.8, 99.7) cm, rotated 1.9 degrees, close to the x 90-290, y 0-200 extent Maboudi's notebook bins over |

The sampling rate (57.43 Hz) is **inferred** from the file. The units check
refuses a session whose largest span is not within 20% of the stated 200 cm.

## Watch out

- **One session of eight.** Nothing here generalizes across rats or days.
- **Which rat and day is inferred.** The file says only Rat1, Day1. The 212
  units match the paper's rat 1 day 1, and the 322 windows match the
  `n_SWRs = 322` that Krause & Drugowitsch (2022)'s code lists for it.
- **How the windows and labels were made is not documented in the file.**
  Krause's code reads them as Pfeiffer's `RippleTimes` and
  `SignificantRipples`. Maboudi et al. (2018) report "27.3% (815 of 2980, n = 8
  sessions)" as P&F 2015's trajectory events, and P&F 2015 reports 815
  open-field trajectory events, so the 110 are most likely 2015's (posterior
  mean steps under 50 cm, 80 cm start to end). They are not P&F 2013's 274
  trajectory events for R1 D1.
- **No LFP**, so neither paper's ripple detector can run here.
- **Missing position samples.** 186 position intervals are longer than two
  median intervals (13.7 s in all, the longest 0.607 s), and the intervals
  vary (7.8 to 25.4 ms), so speed from neighbouring samples is noisy. They are
  kept as stored.

## Checks against the author's own output

**Detection.** `ripple_detection.literature_methods.pfeiffer_2013`, the
paper's population-burst rule, run on the 203 excitatory units' spikes at
1 kHz with the recording's speed, finds 402 candidates. They overlap 272 of
Pfeiffer's 322 windows (84.5%), and 291 of the 402 overlap one of his
(72.4%); 103 of his 110 significant windows are found. Of the 50 windows it
misses, 30 never rise 3 SD above the stopped mean of the smoothed population
histogram, and 26 reach 5 cm/s on our speed: they are not population bursts,
which fits windows found from the ripple envelope. `pfeiffer_2013_ripples`
and `pfeiffer_2015` need that LFP, which the file does not hold, and no proxy
is made from spikes.

**Judging.** Each of the 322 windows is decoded with place fields of the 203
excitatory units (2 cm bins, 4 cm smoothing, speed above 5 cm/s; every unit
peaks above 1 Hz) in 20 ms windows advanced 5 ms, then judged by the
package's presets, unmodified:

| Preset | Judged | Both | Ours only | His only | Neither | Agreement | Kappa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `PFEIFFER_FOSTER_2015` | 78 | 65 | 13 | 45 | 199 | 82.0% | 0.569 |
| `PFEIFFER_FOSTER_2013` | 94 | 40 | 54 | 70 | 158 | 61.5% | 0.113 |

The 2015 rule fits his labels far better than the 2013 rule, as expected if
they are 2015's. Its misses are of one kind: 43 of the 45 windows he labels
and it rejects decode here to a path shorter than 80 cm (a median 55.4 cm
between the first and last posterior means). Their posteriors are broader
(a median 0.059 of the mass in the peak bin, against 0.073 where we agree),
which pulls a mean toward the arena's centre, but that is not all of it: over
the same segment the most likely (MAP) position runs a median 74.4 cm, and
reaches 80 cm in only 16 of the 45. Smoothing the rate instead of the counts
and occupancy changes nothing (82.3%, kappa 0.572). Neither paper says how
unvisited bins, speed or smoothing were handled, and no code was published, so
what made his decoded paths longer is not known. Both presets follow their
papers' Methods as written, and their notes give these numbers.

## Worked result

**Status: not possible yet.**

The test, fixed in writing before any window was decoded: `PFEIFFER_FOSTER_2015`
agrees with Pfeiffer's labels on at least 80% of the 322 windows and Cohen's
kappa is at least 0.6. Agreement is 82.0% but kappa is 0.569, so it is not met.

P&F 2013 reports 144 to 373 trajectory events per session (25.3% to 43.9% of
candidates), 274 for R1 D1. Those come from its own population-burst
candidates under its own rule, so neither the count nor the share can be set
against the file's 110 of 322 (34.2%).

## Cite

Pfeiffer BE, Foster DJ (2013). Hippocampal place-cell sequences depict future
paths to remembered goals. Nature 497, 74-79.
<https://doi.org/10.1038/nature12112>

Data: Maboudi K, Ackermann E, de Jong LW, Pfeiffer BE, Foster D, Diba K,
Kemere C (2018). Uncovering temporal structure in hippocampal output patterns.
eLife 7, e34467. <https://doi.org/10.7554/eLife.34467>; sample recording
`data/openfield.nel` in their code repository, as archived by eLife.
