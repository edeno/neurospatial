# Carey et al. (2019): rat CA1 replay on a food-or-water T-maze

**The question.** When a rat's motivation changes from food to water, does
hippocampal replay move toward the outcome it now wants? These are tetrode
recordings from dorsal CA1 of four rats over six days each, on a T-maze with
food on the left and water on the right, with long rests before, during and
after the task. The rat was restricted of food or of water overnight, so the
same maze had a different value each day. The session suits replay on two
trajectories that share a stem, and detection tested against the authors' own
events.

| | |
| --- | --- |
| Dataset id | `carey_2019` |
| Animal and region | Male Long-Evans rats R042, R044, R050, R064; dorsal CA1 |
| Apparatus | Elevated T-maze, a stem and two arms of about 334 cm from start to reward (257 cm for R042, which had no start platform yet), with rest pedestals beside the arms |
| Archive | DataLad `labs/mvdm/MotivationalT` (ODC-PDDL-1.0): one folder of loose files per session, about 240 to 300 MB each to load, 20 MB per LFP channel on request |
| Sessions | 24, the paper's 19 and five it left out. One is checked by hand: the default `R050-2014-03-29` (72 units, 30 rated good, 42 questionable) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'carey_2019'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("carey_2019")
rates = compute_spatial_rates(
    rec.env,
    rec.spike_times,
    rec.times,
    rec.positions,
    epochs=rec.epochs["matched_left"],
    min_speed=rec.params["min_speed"],
    method="binned",
    bandwidth=rec.params["bin_size"],
)

samples, times = rec.lfp.fetch(rec.lfp.ripple_channels[:1])  # the paper's channel
```

## What was recorded

Each day had three parts. The rat rested on a terracotta flower pot for 20 to 30
minutes (`rec.epochs["prerecord"]`), ran 40 minutes of the task (`"task"`), and
rested 5 to 15 minutes more (`"postrecord"`). In the task it ran from the stem
to the choice point and turned left for food pellets or right for sucrose water,
then was moved to a pedestal beside that arm until the next trial
(`"rest_left_pedestal"`, `"rest_right_pedestal"`). A barrier at the choice point
forced some trials down the unfavoured arm, so that place fields could be
estimated on both.

`rec.trials` has `arm` (`L` or `R`) and `bad`. `rec.events` holds the photobeam
and dispenser events, named as the lab's `getEvents_Tmaze.m` names them, and
`hand_labelled_swr`, the 50 ripples (150 in one session) someone marked by hand,
80 ms each. `rec.params["restriction"]` says what was withheld that day,
`"layout"` where the food was, and `"in_paper"` whether the paper used the session.
`rec.epochs["matched_left"]` and `["matched_right"]` are equal numbers of each
side without the bad trials, the paper's rule.

Sources read: the paper and its supplement (Nature Neuroscience 22, 1450-1459);
the archive's `README.txt` and `datapackage.json`; the files; the paper's code
(github.com/vandermeerlab/papers at 3aebc8e) and the lab's code it pins
(github.com/vandermeerlab/vandermeerlab at ad0bbd4), read and not run.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| spike times: `.t` and `._t` big-endian `uint32` / 10,000, in s | stated | Each file's header: tenths of a millisecond. The two extensions are `units["quality"]`, `good` and `questionable` |
| positions: pixels / `keys.m` `convFact`, in cm | stated | `keys.m`, and the authors' trajectories in pixels and cm differ by exactly these factors |
| frames with x or y 0 are NaN | stated | 0 is the file's own mark for no detection, which the lab's `LoadPos.m` drops by default |
| LFP: stored values x `ADBitVolts` x 1e6, in microvolts, 2 kHz | stated | Each `.ncs` header. On the default session the times equal the paper's candidate file |
| environment: the T-maze from the authors' drawn left and right trajectories, 3 cm bins | stated | `metadata.coord`, against `keys.m`'s lengths |
| `track_width`: twice the 99% quantile of trial samples' distance from the track | inferred | A placement tolerance, 54.7 cm on the default session, 38 to 118 cm over the sessions. No source states the track's width, and trials include moments off it |
| 48 spike files of `R064-2015-04-23` read as 64-bit | inferred | Their headers do not say so, but every other 32-bit word is 0 |

The sampling rate is the header's 29.97 Hz; frames are a median 33.1 ms apart, or
30.2 Hz, on the default session. The stated span is `keys.m`'s `realTrackDims`, 185
cm, within 30%, since the head and the pedestals reach past the maze.

## Watch out

- **Zeros become NaN here.** Other datasets keep samples at (0, 0) as stored. In
  this format 0 means no detection, so frames with a 0 are NaN: 3.7% of the
  default session.
- **Most of a session is off the track.** The rat rests on a pot or pedestal 30 to
  50 cm from the track, so 62.8% of the default session's tracked samples fall in
  no bin. `rec.env` is the track only; the summary's generic words about head
  spread do not apply to these samples.
- **Questionable units.** `._t` clusters are the lab's rating of 5, "questionable".
  The paper's code loads them for detection and decoding, and its "at least 40
  cells" needs them. Filter on `units["quality"]` if you want the others.
- **Which ripple channel.** `keys.m` lists `goodSWR`, best first, but in 3 of the 24
  sessions, this one among them, the paper's candidate file names another of
  them. `rec.lfp.ripple_channels` starts with the candidate file's channel.
- **Mirrored plots.** Positions are on the camera's axes, whose y runs down the
  image. With y up, a plot puts the left (food) arm on the rat's right.
- **No headings.** The tracker's angle field does not follow movement: minus the
  direction of movement it has a median spread of 74 degrees.
- **The LFP lags the spikes by about 2 ms.** The headers state a 1984 us filter
  delay with compensation disabled. Neither the lab's code nor the loader
  removes it.
- **R042's spikes are cut.** The spike files leave out the times the rat ate;
  `rec.epochs["spikes_kept"]` holds what is left, so a silence outside it is not
  silence. R042 was also recorded referenced to ground, and its stem is shorter.
- **R044's headstage came off.** `rec.epochs["headstage_attached"]` holds the
  attached time, and in three sessions `"headstage_attached_by_hand"` the file
  the paper's detection also used. The paper kept only two of its six sessions.
- **A session is about a hundred files.** They are fetched one at a time. The
  archive is one server, so do not fetch from it in parallel.

## Analysis choices

| `rec.params` | Value | Source |
| --- | --- | --- |
| `bin_size` | 3 cm | the paper: "approximately 3 cm per bin", for `rec.env`. `trajectory_environment` cuts a trajectory into 111 equal bins, as the paper's code does: about 3 cm, 2.3 cm on R042's shorter paths |
| `min_speed` | 5 cm/s | the run gate of the paper's code; its text says about 4 cm/s. Its code's detection gate is 10 pixels/s, 3.4 to 4.2 cm/s by axis and rat |
| `pathlength` | from `keys.m` | the length the paper's code scales each trajectory to, cm |
| `dt` | 0.25 s | our choice, for held-out decoding of running |
| `analysis_epochs` | the trials | |
| `train_epochs`, `test_epochs` | even and odd trials | held-out decoding |

## Validation against the paper's events

The paper's code ships its candidate events, one file per session, and each
stores the paper's joint score sample by sample. Its repository's license speaks
of software and not of data, so the files are not redistributed; a test downloads
one. So detection can be checked before any event logic.

Run on the default session, `CSC03a`, with the 50 hand-labelled SWRs:

| Step | Result |
| --- | --- |
| Our joint score against the stored one | r = 0.9999997; 99.99% of samples above 4 agree |
| `literature_methods.carey_2019` as packaged | 1,328 events against the paper's 1,654; 73.5% of the paper's overlap ours, 91.6% of ours overlap the paper's |
| With the paper's theta channel only | 1,363; 78.3% and 95.0% |
| With the paper's speed only | 1,632; 92.9% and 94.1% |
| With both | 1,678; 98.6% and 97.2% |

The score matching says the data are read right and that these 50 examples build
the published template. The gap is in two inputs, not in the score:

- **Speed, the larger part.** The paper's `dxdt.m` picks an adaptive window up to
  1 s, then smooths by 0.5 s, and thresholds 10 pixels/s. The session speed given
  to the detector is smoothed by 0.1 s, against 4 cm/s, so it crosses the
  threshold more often: of 4,628 events above the score threshold, 1,974 survive
  it against 2,516.
- **Theta channel.** `ripple_detection`'s `carey_2019` filters theta from the
  ripple channel; the paper's code uses `goodTheta`, here the reference channel
  `HS1R2`.

Nothing was tuned. Use the packaged method as it is and expect fewer events, or
pass the paper's two inputs to `Carey_candidate_detector` as
`tests/datasets/test_carey_2019.py` does.

## Worked result

**Status: sanity property (level B). Its fixed criterion is not met.** A property of
the detection and decoder, not a number of the paper, which states no count for one
session.

The analysis follows the paper's code where the code and the text differ, since the
code made the published numbers.

| Step | The paper's text | Here, from its code |
| --- | --- | --- |
| Run gate for rate maps | about 4 cm/s | 5 cm/s |
| Poisson exposure | not stated | 5 ms for the 25 ms counts, as `DecodeZ.m` |
| Position bins | about 3 cm | 111 over each drawn path: 2.76 and 2.86 cm on this session, 2.3 cm on R042 |
| Largest step of a sequence | 40 cm | 13 bins, 17 for R042 |

The whole session is counted in 25 ms windows stepped by 5 ms, on each trajectory's
own units. Windows with at least 4 active cells are decoded, and `replay_segments`
finds sequences of at least 10 of them within each contiguous run, with the
preset `CAREY_2019`'s rules in the code's bins. Then, in the
code's order, each side keeps sequences that overlap a candidate and last at least
50 ms, and loses any that overlap the other side's. Two steps are not applied: the
paper's identity-shuffle filter, about which we make no claim, and the 2 ms Gaussian
spike density, because `decode_position` takes whole counts.

The criterion and both nulls were fixed in writing for events, before any decode.
The count moved to sequences before this count was seen, with the same draws and
seeds. Event counts of an earlier route at 4 cm/s had been seen by then.

| | Sequences | 95th percentile, 20 draws |
| --- | --- | --- |
| Our 1,328 candidates | 43 (11 left, 32 right) | |
| Maps fitted with each unit's spikes shifted | 17 to 57, median 34 | 45.6 |
| Decoded spikes all shifted by one offset | 0 to 8 | 8.0 |
| The paper's 1,654 candidates | 47 (13 left, 34 right) | 57.65 and 8.05 |

The criterion is not met. The count exceeds the second null and not the first, on
our candidates and on the paper's. The first null keeps each event's burst and
breaks only which place each cell codes, so a count of this size is what these fixed
rules give without that code: a 50 ms sequence is about two independent 25 ms
windows. The old count, candidates that a sequence overlaps before the two sides are
compared, is 59 here, above both 95th percentiles for that count, 51.85 and 10.05.
The criterion now asks about sequences, the count the paper makes.

Against the paper: Supplementary Fig. 6 gives 429 sequences for R050's six sessions,
with no figure per session. Here the 43 are 11 left and 32 right, a left share of
0.26 on a food-restricted day, where the paper's mean over food-restricted sessions
is 0.35. The first null gives a median of 0.34, so this one session says nothing
about that finding.

**What the number can say.** We planted replay into each of the 1,328 candidates,
once on each trajectory. Every spike kept its time. Within the candidate, the cell
of each kept unit's spike was drawn from that side's rate maps along a real running
path played 15 times faster. Each was then judged by the route above.

| Planted into the 1,328 candidates, each side | Judged replay |
| --- | --- |
| Sweep along a path | 152 of 2,656, 5.7% |
| Stays in one place | 173 of 2,656, 6.5% |
| No position content | 85 of 2,656, 3.2% |

Before the two sides are compared the sweep is 9.5%: a planted sequence that also
decodes on the other side is removed. Recall depends on length: 0.2% of sweeps under
60 ms are judged (3 of 1,440), 6.9% from 60 to 90 ms, and 23.7% at 90 ms or more (92
of 388). The path stays inside the candidate, the median candidate is under 60 ms,
and the rule needs 50 ms of it, so a real sequence that runs past its candidate's
ends is not what this measures. This judge passes about one known replay in twenty
at these spike counts, and it passes a stationary event as often. The real
candidates' 3.2% is no more than events with no content pass. It is not a rate of
replay, and a low share is not evidence that replay is absent. This is a limit of
the number, not a measurement of these rats: planted events come from the model the
decoder assumes, and compression 15 is a choice.

### Across the paper's 19 sessions

An exploratory run, with no criterion: the same analysis on the paper's own
candidates in each of its sessions. It was written down before those sessions ran,
and moved to this route before these counts were seen. `carey_2019.ACROSS_SESSIONS`
holds the numbers, and a test pins the observed ones.

| | R042 | R044 | R050 | R064 |
| --- | --- | --- | --- | --- |
| Sequences here | 1,142 | 86 | 369 | 1,723 |
| The paper's sequences (Supplementary Fig. 6) | 1,257 | 95 | 429 | 1,999 |
| Events with a sequence on one side, the old count | 1,093 | 85 | 360 | 1,671 |

In all 3,320 against the paper's 3,780, without the two steps named above. With each
unit's spikes shifted before fitting the maps, the same total is 2,554 to 3,081 over
20 draws, median 2,771. So it is above every draw, but most of it is what the rules
find anyway. Three of the 19 sessions pass the 95th percentile of their own null.

The left share looks like the paper's main finding and is not a reproduction of it.
It is 0.37 on food-restricted days and 0.59 on water-restricted days (the paper: 0.35
and 0.57), z = -2.69 with the labels permuted, and it correlates -0.68 with the share
of left trials (the paper: -0.54). But the shifted maps give the same difference: a
median of -0.251 against the observed -0.220, and 12 of 20 draws reach it. The left
share follows the share of kept units on the left (r = 0.90; 44% on food days, 54% on
water days), and this analysis keeps units per side as the paper's code does. Why the
kept units follow the restriction was not established. So this analysis cannot tell
that shift from a bias of the decoder.

All of this is measured on our analysis. This page does not assess the paper's own
finding.

## Cite

Carey AA, Tanaka Y, van der Meer MAA (2019). Reward revaluation biases hippocampal
replay content away from the preferred outcome. Nature Neuroscience 22, 1450-1459.
<https://doi.org/10.1038/s41593-019-0464-6>

Data: van der Meer, M. (2017) MotivationalT: extracellular tetrode recordings from
the dorsal CA1 area of the hippocampus in behaving rats [Data set]. DataLad.
<http://datasets.datalad.org/?dir=/labs/mvdm/MotivationalT>
