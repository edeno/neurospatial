# Bhattarai et al. (2020): rat CA1 replay in a spatial sequence memory task

**The question.** How do reward and recent navigation change awake forward and
reverse replay? Three rats ran one of four spatial sequences to one reward site
while tetrodes recorded dorsal CA1. The archive holds the 25 session sets the
paper analysed: spikes, head position and direction, trials and one LFP channel.
It suits place fields per trajectory, decoding along a linear path, and a first
look at candidate events on a task with blocks and reward. **The paper's replay
result cannot be reproduced from the file's LFP**; see the worked result below.

| | |
| --- | --- |
| Dataset id | `bhattarai_2020` |
| Animal and region | Three male Sprague Dawley rats, dorsal CA1 |
| Apparatus | A loop of about 3 m with a figure 8-shaped section of two rings; sequences RR, LL, RL and LR to one reward site; sliding doors and a 3 s delay |
| Archive | Figshare 10032866, version 2 (CC0), `DataSet.zip`, 659 MB |
| Sessions | 25 sets, named as in the paper's Table S1. One is checked by hand: the default `rat2_set2` (42 units) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'bhattarai_2020'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head()
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import bhattarai_2020, load

rec = load("bhattarai_2020")          # fetches 50 MB of the 659 MB zip
trials = rec.trials.intervals_table()
block = trials[(trials["session"] == 1) & (trials["block"] == 1)]
template = block[block["forced"] | block["rewarded"]]

# The paper's rate maps: one linear path per sequence, above 4 cm/s
track = bhattarai_2020.sequence_track(block["sequence"].iloc[0])
rates = compute_spatial_rates(
    track,
    rec.spike_times,
    rec.times,
    rec.positions,
    method="binned",
    bandwidth=rec.params["bandwidth"],
    min_speed=rec.params["min_speed"],
    epochs=template[["start_time", "stop_time"]].to_numpy(),
)

samples, times = rec.lfp.fetch()      # 1.08 GB, once, for all 25 sets
```

## What was recorded

Each day a rat ran two consecutive sessions, together a *session set*. A session
has four blocks in random order, one per correct sequence, each of 5 forced-choice
trials followed by 15 free-choice trials. A trial starts as the rat leaves the
reward zone, 30 cm from the nozzle. It runs along the return arm into the delay
zone, waits 3 s, runs through both rings and reaches the reward site, where 30 uL
of water is delivered for a forced or a correct trial. Blocks are named for the
rat's turn at each ring: `RL` is right at the first ring, then left at the second.

The 15 tetrodes were lowered into the CA1 cell layer. The head position comes
from a set of red and green LEDs on the headstage at 30 Hz. The authors recorded
28 sets and left out three whose decoding error exceeded 20 cm. Table S1 lists
the 25 by rat (6, 9 and 10 sets), and the archive's cells are in its order.

`rec.trials` has one row per trial with `session`, `block`, `trial_in_block`,
`sequence`, `choice` (what the rat ran), `forced`, `rewarded`, `delay_end`,
`reward_start` and `end_known`. `rec.epochs` holds `session_1`, `session_2`,
`forced_choice`, `free_choice` and `reward`, from the start of reward to the end
of the trial. There are no rest or sleep epochs: the files hold none.

Sources read: the paper (Bhattarai, Lee & Jung 2020, PNAS 117:689-697), its SI
Appendix (Materials and Methods, Fig. S1, Table S1), the archive's `MetaData.docx`
and the files. `lfp.mat` is MATLAB 7.3 (HDF5); the others are MATLAB 5.0.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| positions: cm = (x, -y) in `VT.mat` pixels / 3.6 | inferred | Nothing read states a unit. The linearized maze runs 0 to 300 cm (Fig. 3A), and a trial's path on the graph here is 304 to 308 cm. A trial starts 30 cm from the nozzle (SI Appendix); the head moves a median 103 pixels from the start of reward to the next trial's start. The arms of a ring lie about 10 cm either side of their middle, as in Fig. S1B. |
| the y axis negated | inferred | Read with y up, the path runs the opposite way round from Fig. S1A under any rotation, the blocks called LL take the arms on the rat's right, and the stored heading is a bearing only if y points down the image. |
| headings: pi/2 - radians(`VT.mat` column 4) | inferred | `MetaData.docx` says degrees, no frame. As stored the values are a clockwise bearing from the camera's up. Converted, their mean difference from movement while running is -0.3 to +13.7 degrees over the sets, with a median spread of 10 to 18 degrees, and 41 and 47 degrees in `rat2_set5` and `rat2_set7`, whose tracking leaves the maze. |
| headings: NaN where the file holds exactly 0 | inferred | 45% to 83% of a set's samples hold exactly 0; the values next to it hold under 0.1%. |
| spike times as stored, s, sorted | stated | `MetaData.docx`. |
| `units["label"]`: `complex-spike` below 5 Hz with a burst index above 40%, else `theta` | stated | The SI Appendix's rule. It gives the paper's 648 of 698 units over the archive. |
| trials: `events.mat` rows with `beh.mat` rows at a fixed shift | inferred | The tables are on two clocks. At the shift, the start of reward equals `beh.mat`'s last sensor plus one constant, within 25 ms in most trials. The shift is 0 in 23 sets, 1 in `rat3_set10` and 8 in `rat3_set7`, after an aborted block. |
| session ids from the order of the cells | inferred | The share of rewarded free-choice trials follows Table S1's Performance column row by row, within 0.8 points, and 1.8 in `rat3_set7`. |
| environment: a track graph read from the samples | inferred | The paper gives no ring dimensions. The waypoints fit the samples of all 25 sets, which share one camera view; bins are 2 cm. |
| `track_width`: twice the samples' 99% quantile distance from the graph | inferred | A placement tolerance, measured per session. |

## Watch out

- **The cm scale is inferred from lengths, not speeds.** The paper's mean speeds
  (track 15.8, return 17.5, delay 12.2, reward 7.3 cm/s) fit no single scale: at
  3.6 pixels per cm the head averages about 40 cm/s from the end of the delay to
  the reward and 5 cm/s in the reward period. x and y share one scale, and nothing
  read says the pixels are square.
- **One LFP channel, not two.** The paper averaged ripple power over one channel
  each of two tetrodes. `lfp.mat` holds one channel per set, and the files do not
  say which tetrode it is. The paper's theta power came from a tetrode above the
  cell layer, so it may be that one.
- **The LFP is saturated.** Its values stop at +/-327.67, and 5% to 56% of a set's
  samples sit there (21.4% in the default set). On a rail the ripple band is flat,
  and each step off a rail adds broadband power. The unit is not stated.
- **Ripple power marks population events weakly.** Around peaks of 100-250 Hz power
  the rate of all units is 0.65 to 1.59 times its level 120 to 200 ms away, over the
  25 sets (1.52 in the default set).
- **No rest.** Tracking starts at most 34 s before the first trial and ends at most
  101 s after the last. The two sessions of a set follow each other. So there is
  no pre-task baseline, and a threshold's mean and SD come from the task itself.
- **Head direction is often missing.** 45% to 83% of samples hold no direction.
- **Samples at (0, 0) stay.** They are lost tracking, 0.1% to 2% of samples, and are
  kept as stored. Set them to NaN before analysis. Heading is NaN there too.
- **Off-maze samples.** In `rat2_set5` and `rat2_set7`, over 6% of samples lie more
  than 30 cm from the track, in a part of the camera's view with no maze. These
  sets have the highest head-direction coverage too, so this may be a reflection.
- **Lost trial starts.** In `rat2_set5` to `rat2_set9`, `rat3_set1` and `rat3_set2`,
  some `events.mat` rows start at their own end. That trial starts at its end of
  delay here, the trial before ends there with `end_known` False, and it is left
  out of `epochs["reward"]`. The ends of delay and starts of reward are not affected.
- **`rat3_set3` fails the clock check.** Five of its units hold exact zeros among
  their spike times (3,990 spikes). They are kept, so load it with
  `accept_failed_checks=("clock",)` and use `train[train > 0]`.
- **`MetaData.docx` is one column off** in `beh.mat` from column 7. Column 7 is the
  session's count of correct trials, 8 to 11 the error sequences, 12 and 13 the
  forced trials and the delay. `rec.trials` reads them that way.
- **The last trial of session 1.** In 18 sets its start of reward in `events.mat`
  is 1 to 57 s after `beh.mat`'s reward sensor.
- **Sessions repeat members.** Every set reads the same four small members, cached
  once per session (50 MB each). `lfp.mat` is cached once for all.

## Analysis choices

| `rec.params` | Value | Source |
| --- | --- | --- |
| `bin_size`, `bandwidth` | 2 cm, 6 cm | the paper's rate maps (SI Appendix) |
| `min_speed` | 4 cm/s | the paper's rate maps exclude "periods of immobility (speed < 4 cm/s)" |
| `dt`, `time_step` | 0.02 s, 0.01 s | the paper's replay decoding bins |
| `analysis_epochs` | first trial's start to last trial's end | all the task |

`bhattarai_2020.sequence_track(sequence)` builds the 1-D path of one sequence,
from the start of a trial to the reward site. The paper projects positions onto a
linear path and decodes each block on its correct path; this is that track.
`neurospatial.replay.methods.BHATTARAI_2020` is the paper's replay test.

## Worked result

**Status: not possible yet.**

The finding to reproduce: about half of the candidate replay events are significant
replays of the current trajectory. The paper: "We found 2,573 candidate replay
events that are related to the current trajectory (correct path in the current
block)" (Results); "An event was considered as a significant replay if P < 0.05
(n = 1342; 52.2% of candidate replay events" (SI Appendix). Table S1 lists each
set: 278 candidates and 153 replays for `rat2_set2`, and its columns do sum to
2,573 and 1,342.

**Why it is not possible yet.** A candidate must coincide with an SWR detected on
two tetrodes, and the archive holds one channel, saturated in part, whose ripple
power rises little at population bursts.

**The attempt, with the paper's recipe and preset.** On the default set:

- complex-spike units by the paper's rule; for each block, binned rate maps on
  `sequence_track` (2 cm, bandwidth 6 cm, above 4 cm/s, forced and correct
  free-choice trials);
- place cells: units `detect_place_fields` finds a field in, our choice as the
  design states, not the paper's three criteria. Each set's union is close to
  Table S1's count (36 against 31 here; 550 against 497 over the sets);
- candidates: `ripple_detection.literature_methods.bhattarai_2020` at commit
  0eddae64, power as the squared signal, the last spike of the 300 ms window as the
  event's end, each block's place cells, within the reward period of its
  free-choice trials;
- `bhattarai_2020_ripples` refuses fewer than two channels, so the file's channel
  fills both columns. Their mean is that channel's own power; no second signal is
  made up;
- decoding in 20 ms windows stepped by 10 ms, judged with `BHATTARAI_2020`.

The criterion, fixed before any null ran: the share of candidates judged replay
exceeds the 95th percentile of the same share with cell identities permuted within
each block (20 draws, seed 2026). **It was not met.**

| | Candidates | Replays | Share | Permuted cells, 95th percentile |
| --- | --- | --- | --- | --- |
| Paper, `rat2_set2` (Table S1) | 278 | 153 | 55.0% | |
| Ours, the paper's recipe | 37 | 9 | 24.3% | 27.0% |
| Ours, the same bursts with no SWR requirement | 534 | 95 | 17.8% | 15.9% |
| Paper, all 25 sets | 2,573 | 1,342 | 52.2% | |
| Ours, the paper's recipe, all 25 sets | 187 | 34 | 18.2% | not run |
| Ours, no SWR requirement, all 25 sets | 5,019 | 788 | 15.7% | not run |

Only the second row is the attempt. The others are the paper's numbers and
diagnostics, stated beside it and not part of the criterion. Where the difference
comes from:

- **The SWR gate.** The paper counts 132,665 SWRs over the sets. Three SD on this
  channel gives 24,028 threshold crossings, and 505 once an SWR must hold five
  place cells. The bursts themselves are in the paper's range: over the sets they
  correlate with Table S1's candidates at r = 0.81 (319 against 313 on `rat2_set1`,
  534 against 278 here).
- **The events are longer.** Ours last 232 ms (the paper: 151.9 +/- 61.5 ms). The
  paper does not say how an event ends.
- **A short event has few shifts.** An event of n time bins has n circular shifts, so
  under 20 time bins, p < 0.05 means no shift fits better. Ours have a median of
  about 11. With permuted cells the test still calls about 15% of events replay at
  alpha 0.05.

The decoder is not the cause: in-sample running error is about 8.5 cm on this set
against Table S1's 5 cm, in 50 ms bins. Nothing was tuned to close the gap.

Level: no published replay number is reproduced. Reproduced at level A: 648 of 698
units are complex-spike cells by the paper's rule, exactly the paper's number.

Source: Bhattarai, Lee & Jung 2020, PNAS 117:689-697, Results; SI Appendix, "SWR
events", "Replay events" and Table S1.

## Cite

Bhattarai B, Lee JW, Jung MW (2020). Distinct effects of reward and navigation
history on hippocampal forward and reverse replays. *Proceedings of the National
Academy of Sciences* 117(1):689-697.
<https://doi.org/10.1073/pnas.1912533117>

Bhattarai, Baburam; Lee, Jong Won; Jung, Min Whan (2019). Supplementary data:
Distinct effects of reward and navigation history on hippocampal forward and reverse
replays. figshare. Dataset. <https://doi.org/10.6084/m9.figshare.10032866.v2>

License: CC0, as the archive states it. Data are fetched from Figshare, not
redistributed.
