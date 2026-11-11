# Ólafsdóttir et al. (2016): CA1 and deep MEC on a Z-track, then rest

**The question.** When a rat rests after running a track, do grid cells in the
deep layers of the medial entorhinal cortex replay the track together with
hippocampal place cells? These are tetrode recordings from dorsal CA1 and MEC
layers V/VI of six rats on the Axona system: each day, 20 or more laps of an
elevated Z-shaped track, then about an hour and a half in a small rest
enclosure, then a foraging session in a 1 m open field. The paper's replay
events all come from the rest. Every detected spike keeps its waveform, so the
data also suit clusterless decoding.

| | |
| --- | --- |
| Dataset id | `olafsdottir_2016` |
| Animal and region | Male Lister Hooded rats R2142, R2192, R2217, R2335, R2336, R2337; dorsal CA1 (tetrodes 9 to 16) and MEC layers V/VI (tetrodes 1 to 8), the other way round for R2142 |
| Apparatus | Elevated Z-track, two parallel 190 cm runways joined by a 220 cm diagonal, 10 cm wide; a cylindrical rest enclosure, 18 cm across and 61 cm high, outside the track's curtains; a 1 m open field |
| Archive | Zenodo 5566548 (CC-BY-4.0): one stored zip of 9.91 GB. A load reads its members by byte range, 0.3 to 1.2 GB a day; the 4.8 kHz LFP is on request |
| Sessions | 10 recording days, one session each. One is checked by hand: the default `R2336-2015-11-01` (59 units: 56 in CA1, 3 in MEC) |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'olafsdottir_2016'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]]
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import load

rec = load("olafsdottir_2016")
outbound = compute_spatial_rates(
    rec.env,
    rec.spike_times,
    rec.times,
    rec.positions,
    epochs=rec.epochs["outbound"],
    min_speed=rec.params["min_speed"],
    method="binned",
    bandwidth=rec.params["bandwidth"],
)

# Every detected spike on tetrode 10 during the rest, with its peak on each wire.
times, amplitudes = rec.marks.select("T10", rec.epochs["rest"])

samples, lfp_times = rec.lfp.fetch(("T9.1",))  # a CA1 wire, 4.8 kHz, microvolts
```

## What was recorded

A recording day has two or three trials, each its own set of Axona files: the
track (`track1`), the rest (`sleepPOST`) and, on some days, the open field
(`screening`). The loader puts them on one clock, the track's, so
`rec.epochs` holds `"track"`, `"rest"` and, where present, `"open_field"`. The
rat started each track session at the same end; `rec.trials` holds its runs
from one end to the other, `direction` `"outbound"` from the start end, and
`rec.epochs["outbound"]` and `["inbound"]` the same rows. The paper built
separate rate maps for the two directions. `rec.epochs["corner_stops"]`,
`["engaged"]` and `["disengaged"]` are the rat's stops at the Z's two corners and
the 2017 paper's sections of them (see
[Awake replay at the corners](#awake-replay-at-the-corners-2017-paper)).

`rec.positions` and `rec.env` are the track only. The rest and the open field
were tracked in other windows of the camera's image, so their positions are in
`rec.params["rest_times"]`, `["rest_positions"]` and `["open_field_..."]`, cm in
their own window. The open field is where the paper classified MEC cells as
grid cells; its cuts cover only some tetrodes (see Watch out).

`rec.marks` holds every spike that crossed the detection threshold on every
tetrode, in every trial, whether or not it was sorted into a cluster: its time,
its largest sample on each of the four wires in microvolts, and its cluster.
`rec.marks.waveforms("T10")` reads the full waveforms (50 samples per wire at
48 kHz) from the cached files. These are the inputs of a clusterless decoder.

Sources read: the paper, its Online Methods, Supplementary Information and
Supplementary Software (Nature Neuroscience 19, 792-794); Ólafsdóttir, Carpenter
& Barry 2017 (Neuron 96, 925-935), on the same rats and track; the deposit's
`README.txt` and record description; Axona's "dacqUSB Data File Formats"
document; and, for what that leaves out, three open readers, read and not run:
ephysiopy's `axonaIO.py` (MIT), the Barry lab's `barrylab.io.axona` (MIT) and
Neo's `axonarawio.py` (BSD-3).

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| spike times: each tetrode file's big-endian 4-byte timestamps / 96,000, plus the trial's start | stated | Axona's format document; each file's `timebase 96000 hz` |
| trial starts: the rest's and open field's `.set` `trial_time` minus the track's | inferred | each file's clock starts at 0; `trial_time` is to the second, and the trials do not overlap |
| units: clusters 1 and up of each trial's Tint `.cut`, joined by id across the day's trials | inferred | cluster 0 is Tint's unassigned spikes; each trial was cut on its own, and the paper decoded rest spikes with track rate maps. 48 of 56 clusters sorted in both trials have a rest waveform nearest the same id on the track |
| marks: each wire's largest stored sample x `1000 ADC_fullscale_mv / (gain 128)`, microvolts | stated | the format's 8-bit samples; the gain of each wire in the trial's `.set`, as ephysiopy scales tetrode waveforms |
| positions: the big LED's pixels x 100 / `pixels_per_metre`, cm; 1023 is NaN | stated | the `.pos` header (300); 1023 is the format's mark for a spot not found |
| times: frame k (from 1) at k / 50 s | stated | Axona's document: "the Xth sample in the file was taken at time X/50 seconds", the frame counter "should be ignored" |
| environment: a Z of three straight pieces fitted to the track's running samples, 2 cm bins | inferred | no file places the track in the camera's frame. The runways are total-least-squares lines through samples moving along them, the corners their tips nearer the diagonal |
| LFP: stored little-endian int16 x `1000 ADC_fullscale_mv / (gain 32768)`, microvolts, 4.8 kHz | stated | the gain of the EEG slot's recording channel in the trial's `.set`, as the Barry lab's reader converts |
| LFP times: sample k (from 1) of a trial at its start + k / 4800 s | inferred | the format document's rule for positions; it gives none for `.egf` |

## Watch out

- **Lengths are about 12% longer than the stated apparatus.** At the header's
  300 pixels per metre the fitted Z is 674 cm long against the stated 600 cm, and
  the 1 m open field spans 118 to 130 cm. The lab's code converts with the same
  header value, so the paper's 2 cm bins were most likely in these units.
- **No headings.** The second, little LED is found in 11% of track frames, and
  where both are found their direction minus the direction of movement has a mean
  of +111 degrees. `rec.params["little_led_positions"]` keeps it.
- **Track only in `rec.positions`.** See What was recorded.
- **A unit joins one cluster id across trials.** Each trial was cut on its own.
  On the default session 8 of 56 clusters sorted in both the track and the rest
  have a rest mean waveform nearer another cluster's on the track.
- **Unsorted tetrodes.** Some tetrodes have no `.cut` in a trial; their spikes are
  in `rec.marks` with cluster -1 and in no unit. On the default session tetrodes
  1, 3, 4, 5, 7 and 8 (MEC) are unsorted, so only 3 MEC units are sorted.
- **Which paper session.** The paper's Supplementary Table 1 numbers each rat's
  sessions without dates. The deposit has 10 days for the paper's 11 sessions;
  R2192 has 2 of its 3, and which is unknown. See [Sessions](#sessions-and-the-papers-table).
- **`R2192-2014-10-01` lost spikes of tetrode 4 in the rest.** Its rest file
  `20141001_R2192_sleepPOST.4` is zeros from inside spike 18,052 of the 63,801
  its header counts to the end, its end marker included, in the deposit itself
  (its SHA-256 is the registry's). The zeros begin on a 4,096-byte boundary and
  cut a spike in half, so the bytes were lost in storage. The load keeps the
  18,051 whole spikes before them and warns. Tetrode 4 is sorted in no trial of
  the day, so no unit loses spikes, but `rec.marks` has no T4 spikes from
  3,410 s to the rest's end at 7,152 s: leave T4 out of that span in a
  clusterless decode, or it reads the lost spikes as no firing.
- **`R2335-2015-10-26`'s second open field has no sorted units.** It
  (`screening2`) starts at 17:26, five hours after the first, and no tetrode has
  a `.cut` for it, so its spikes are in `rec.marks` only.
- **Files the loader leaves out.** `.clu` files (the README: superseded by the
  `.cut` files), `.eeg` (the 250 Hz copy of `.egf`), `.inp`, the `*_info.mat`
  files, and three cut files named for no tetrode: `20140806_R2142_sleepPOST.cut`,
  `20140110_R2192_track1.cut` and `20151127_R2337_screening_6.ut`.
- **Names in the files are not always right.** The R2192 track of 2014-10-01 is
  named `20140110` (the README); the R2217 open fields are named `R2117`; the
  R2217 `.set` comments say "Screening R2168". Trials are paired by folder and by
  their `.set` dates and times, which put the track, rest and open field in that
  order on one morning.

## Analysis choices

| `rec.params` | Value | Source |
| --- | --- | --- |
| `bin_size` | 2 cm | the paper's rate maps |
| `bandwidth` | 10 cm | the paper's Gaussian of sigma 5 bins |
| `min_speed` | 3 cm/s | the paper's rate maps leave out slower periods |
| `end_exclusion`, `corner_exclusion` | 10 cm, 5 cm | the paper leaves out the last 10 cm at either end and 5 cm around each corner |
| `replay_dt` | 0.01 s | the paper's replay decoding bins |
| `dt` | 0.5 s | the paper's check of decoding on the track |
| `analysis_epochs` | the runs | |
| `min_speed_2017` | 10 cm/s | the 2017 paper's rate maps leave out slower periods |
| `immobile_speed` | 3 cm/s | the 2017 paper's replay events stay below it |
| `corner_stop_direction`, `corner_stop_corner` | per corner stop | the run it arrived on, and which corner (1 nearer the start) |
| `train_epochs`, `test_epochs` | even and odd runs | held-out decoding |

## Sessions and the paper's table

The paper analysed 11 sessions of 6 rats (its Supplementary Table 1 gives each
one's place cells, grid cells and "Number of events") and numbers them without
dates. The deposit has 10 recording days. A session's place cells are among its
CA1 units, so a day with fewer CA1 units than a session's place cells is not that
session; that settles four days. Date order does not: it would make
`R2336-2015-11-01` the session with 71 place cells, but that day holds 56 CA1 units.
`rec.params["paper_session"]` holds the match, or `(None, None)`.

| Day | Units (CA1) | The paper's session | Supplementary Table 1 |
| --- | --- | --- | --- |
| `R2142-2014-08-06` | 65 (40) | 1, the only one | 39 place cells, 3 grid cells |
| `R2192-2014-09-17` | 97 (71) | 2 or 3, not 1 | 72, 70 and 68 place cells |
| `R2192-2014-10-01` | 141 (93) | not told | |
| `R2217-2014-12-13` | 108 (64) | not told | 40 and 34 place cells |
| `R2217-2014-12-18` | 139 (40) | not told | |
| `R2335-2015-10-26` | 84 (34) | 1, the only one | 34 place cells, 6 grid cells |
| `R2336-2015-11-01` | 59 (56) | 2: fewer CA1 units than session 1's 71 | 48 place cells, 3 grid cells |
| `R2336-2015-11-04` | 75 (71) | 1 | 71 place cells, 3 grid cells |
| `R2337-2015-11-27` | 109 (72) | 1 | 70 place cells, 3 grid cells |
| `R2337-2015-12-01` | 85 (46) | 2: fewer CA1 units than session 1's 70 | 46 place cells, 3 grid cells |

Units here are every cluster from 1 up in the day's cut files. By the paper's
place-cell rule (with our 5 Hz interneuron cut) the R2336 days have 52 and 59
place cells, and the R2337 days 49 and 43: within each rat the day with more is
the session the table lists with more, though the counts themselves are lower.
On `R2335-2015-10-26`, `R2336-2015-11-04` and `R2337-2015-12-01` the CA1 units
equal the table's place cells exactly (34, 71, 46), and `R2217-2014-12-18`'s 40
equals R2217's session 1, so the table's "place cells" may count the sorted CA1
units. Nothing here relies on that.

## Awake or asleep, track or rest

All of the paper's replay is from the rest, none from the track. Its Methods
detect candidate events "from the rest session", an hour and a half in the rest
enclosure outside the track's curtains, recorded in the light period "to
facilitate rest", and they state no speed, immobility or sleep criterion. The
paper calls the whole period rest and does not separate sleep from quiet waking,
so none of its 1,826 replay events is known to be awake, and none is on the
track. `ripple_detection`'s `olafsdottir_2016` likewise has no speed rule: pass it
the rest alone.

The rest's positions cannot separate sleep from waking either. In the 18 cm
enclosure the head LED's frame-to-frame jitter dominates: on the default session
the apparent speed has a median of 3.6 cm/s and a 95th percentile of 23 cm/s, and
only 1.8% of the rest lies in still stretches of a minute or more by that
measure. Sleep would need the LFP (a theta/delta ratio, say), which the paper did
not use.

So an awake-replay reproduction cannot come from this paper. It comes from
Ólafsdóttir, Carpenter & Barry 2017, on the same track and these rats: see
[Awake replay at the corners](#awake-replay-at-the-corners-2017-paper).

## Worked result

**Not possible yet.** The paper's number is the share of the rest's candidate
events judged replay: "Robust events, with an individual P-value of less than
0.2, were accepted as replay events and submitted to further analyses (1,826 out
of 4,382 putative events)", 41.7% pooled over its 11 sessions. It gives no share,
and no candidate count, per session. Before any analysis of these data we fixed
the criterion: on the default day, the share judged replay lies within 10 points
of 41.7%, from 31.7% to 51.7%.

The route, on `R2336-2015-11-01`:

- **Place cells**, the paper's rule: a CA1 unit with a field of 20 cm or more
  above its mean rate on the track and a peak above 1 Hz, in the outbound or the
  inbound map (2 cm bins, 3 cm/s, a Gaussian of sigma 10 cm, the last 10 cm of
  each end and 5 cm around each corner left out). The paper excludes
  interneurons by "narrow waveforms and high firing rates" with no numbers; we
  chose a mean track rate above 5 Hz before the run. 52 of 56 CA1 units.
- **Candidates**: `literature_methods.olafsdottir_2016` (place-cell multi-unit
  activity in 1 ms bins smoothed by 5 ms, above its mean by 3 SD, bounded at the
  mean, at least 40 ms, at least 15% of the place cells active) on the rest alone,
  at 1 kHz: 1,118 events. CA1 ripple power (150 to 250 Hz, wire T9.1) is 10.6
  times higher inside them than in the rest outside them.
- **Judged** with `OLAFSDOTTIR_2016` as packaged: each event decoded in 10 ms bins
  with the outbound and the inbound maps, each tested against 100 rotations of the
  place cells' maps, the smaller p kept, replay at p < 0.2.

**681 of 1,118 (60.9%) are judged replay, so the criterion is not met.** The gap
is not in neurospatial's line fit: the paper's own Supplementary Software
(`decode_calcPosterior.m`, `lineTraj_decode.m`), ported to a test and run with the
Methods' shuffle on 150 of these events, judges 91 replay where the preset judges
93, 88 by both. Of the code's own choices only its zero padding beyond the track
changes a verdict (3 of 150; the preset gives Davidson's median there); its 1 m/s
speed floor (the Methods say 2), its rule that a line lie on the track for a third
of the event, and its division by the time bins with spikes change none. What
these files cannot settle is which events and cells the paper judged: it gives no
share or candidate count per session (1,118 here against its mean of 398), its
interneuron cut is unstated, and this day holds 56 CA1 units where Supplementary
Table 1 lists 71 or 48 place cells for R2336.

The tests `test_worked_result_share_of_candidates_judged_replay` and
`test_the_papers_code_against_the_preset` in
`tests/datasets/test_olafsdottir_2016.py` pin these numbers.

## Awake replay at the corners (2017 paper)

**The question.** Ólafsdóttir, Carpenter & Barry (2017, Neuron 96, 925-935)
looked at replay on the track itself. The rat stopped at each corner of the Z
for food, and the paper asks whether replay there depends on what the rat is
doing: in the first and last 5 s of a stop ("engaged", just after arriving and
just before leaving) or in the time between ("disengaged"). It reports that
engaged reactivations mostly decode to the direction the rat is running (62.26%,
against 52.19% when disengaged), and that 36.88% of engaged and 38.60% of
disengaged putative trajectory events are replay.

The paper recorded eight rats over 24 sessions, among them the six rats of this
deposit, and says its data are available on request; it lists no sessions, so
that these 10 days are among its 24 is inferred from the rat IDs. Its numbers
are pooled over sessions, so they are compared here with numbers pooled over
the deposit's days.

The loader gives what the analysis needs. `rec.epochs["corner_stops"]` holds
each stay at a corner, from arriving within 20 cm of it (along the track) to
the last sample that near before the rat is 40 cm away; the paper gives no
distance, and on the default day the samples below 3 cm/s near a corner lie
within 20 cm of it. `rec.params["corner_stop_direction"]` gives the run each stop
arrived on. `rec.epochs["engaged"]` and `["disengaged"]` are the paper's
sections: the first and last 5 s of a stop and the time between, a stop shorter
than 10 s engaged throughout, and sections under 2 s left out.

<!-- docs-test: skip needs a download and ripple_detection -->
```python
import numpy as np
from ripple_detection.literature_methods import Recording, run_method

from neurospatial.behavior import compute_speed
from neurospatial.datasets import load

rec = load("olafsdottir_2016")
place_cells = [...]  # your CA1 place cells, by unit id
names = list(rec.units.index)
track = rec.epochs["track"][0]
grid = track[0] + np.arange(round((track[1] - track[0]) * 1000)) / 1000.0
# One row per ms of the track trial: a count belongs to the millisecond it falls in.
multiunit = np.column_stack(
    [
        np.bincount(
            np.searchsorted(grid, train[(train >= grid[0]) & (train < track[1])],
                            side="right") - 1,
            minlength=grid.size,
        )
        for train in (rec.spike_times[names.index(u)] for u in place_cells)
    ]
)
speed = np.interp(grid, rec.times, compute_speed(rec.times, rec.positions))
signals = Recording.from_arrays(
    grid,
    1000.0,
    multiunit=multiunit,
    place_cells=np.ones(len(place_cells), dtype=bool),
    speed=speed,
)
# Events wholly inside an engaged section, the rat below 3 cm/s throughout.
engaged = run_method(
    "olafsdottir_2017",
    signals,
    behavior_intervals=rec.epochs["engaged"],
    analysis="trajectory",  # adds 15% and more than 5 place cells active
)
```

### What was run

Fixed in writing before any candidate event on a track was detected or judged
(the constants above `SEED_2017` in `tests/datasets/test_olafsdottir_2016.py`):

- **Which days.** The paper's rule: a session enters if the median error of
  decoding the running rat in 500 ms bins is at most 30 cm. Here the place
  cells' outbound and inbound maps over all runs decode each run with its own
  direction's maps; the error is the distance along the track from the most
  probable bin to the rat, in bins where it runs faster than 10 cm/s. All 10
  days pass, at 5.2 to 21.7 cm. (`R2192-2014-10-01` entered after the first
  run, once the reader kept the spikes before its zero-filled file: the rule
  was every day the deposit's files can be read for, so the analysis did not
  change.)
  The paper's other exclusion (R2336's day 4, headstages disconnected in the
  rest) cannot be applied: the deposit does not number days, and neither R2336
  rest is short.
- **Place cells**, the paper's rule on its 2017 maps (2 cm bins, Gaussian of
  sigma 10 cm, ends and corners left out, speed above 10 cm/s): a field of 20
  contiguous bins above the cell's mean track rate and a peak above 1 Hz, in
  either direction. Interneurons, "narrow waveforms and high firing rates" in the
  paper, are a mean track rate above 5 Hz, our choice, as in the 2016 result.
- **Candidates**: `literature_methods.olafsdottir_2017` on the track trial at
  1 kHz (place-cell multi-unit activity smoothed by 5 ms, above its mean by 3 SD,
  bounded at the mean, at least 40 ms, the rat below 3 cm/s throughout, by the
  session's speed), kept when wholly inside one engaged or disengaged section.
  `analysis="trajectory"` adds the paper's 15% and more than 5 place cells.
- **Trajectory events judged** by `OLAFSDOTTIR_2017` as packaged: each
  direction's posterior in 10 ms bins scored, the direction with the higher
  score kept with its own p, replay at p < 0.025.
- **Reactivation events** decoded from their summed spikes with each
  direction's maps, as the Methods describe ("two posterior probability
  matrices were generated for each event"); the arm of the six (three arms, two
  directions) with the most posterior is the event's, congruent when its
  direction is the run the rat arrived on. Chance is the paper's cell-ID
  shuffle: each active place cell's spikes given to a silent one, 100 times.
- **Criteria.** A, the worked result: pooled over the days, the share of
  trajectory events judged replay lies within 10 points of the paper's pooled
  38.1% (513 of 1,347; the paper gives 149 engaged and 364 disengaged replay
  events, and 36.88% and 38.60%, so 404 and 943 putative events, inferred).
  B, the paper's main finding: engaged reactivations are congruent more often
  than chance and than disengaged ones, each by the paper's bootstrap (more
  than 97.5% of 10,000 resamples).

### Results

**Criterion A is met: 99 of 325 trajectory events (30.5%) are judged replay**,
inside 28.1% to 48.1%, though near its lower edge.

| Pooled over 10 days | Engaged | Disengaged | Pooled | Paper |
| --- | --- | --- | --- | --- |
| Putative trajectory events | 63 | 262 | 325 | 404, 943, 1,347 |
| Judged replay by `OLAFSDOTTIR_2017` | 19 (30.2%) | 80 (30.5%) | 99 (30.5%) | 36.88%, 38.60%, 38.1% |
| By the paper's rule, at most 2 of 100 shuffles as high | 24 (38.1%) | 94 (35.9%) | 118 (36.3%) | |

As in the paper, the engaged and disengaged shares are close. Two differences
from the paper are measured here:

- **The threshold.** The paper calls an event replay when its score exceeds the
  97.5th percentile of its 100 shuffles, which allows 2 shuffles as high;
  `OLAFSDOTTIR_2017` takes p = (k + 1) / (n + 1) < 0.025, which allows 1. The
  paper's rule calls 19 more events replay, 36.3% of the 325.
- **Fewer events.** 325 putative trajectory events on 10 days, against the
  paper's 1,347 on 24 sessions. The rat must stay below 3 cm/s through an event,
  and by `compute_speed`'s default (0.1 s smoothing) the head LED is that slow
  in 10.6% of the default day's corner stops, since the head moves while the
  rat eats. The paper does not say how it
  estimated speed.

**Criterion B is not met.** Decoded as the Methods describe, engaged
reactivations are congruent in 124 of 282 (44.0%), against a cell-ID shuffle
chance of 51.4% (0.8% of resamples above chance), and disengaged ones in 515 of
994 (51.8%, chance 49.1%).

The cause is the decode, not the data. Each direction's posterior sums to 1 on
its own, so comparing arms across the two cannot say which direction the spikes
match. On the default day's runs, the same rule names the direction the rat is
running in 1,315 of 2,564 bins of 500 ms (51.3%, a coin's rate);
`neurospatial.decoding.decode_jointly`, one posterior over both directions'
maps, names it in 2,084 (81.3%). Decoded jointly, which was not fixed before the
run:

| Reactivation events, pooled | Engaged | Disengaged |
| --- | --- | --- |
| Congruent | 179 of 282 (63.5%) | 601 of 994 (60.5%) |
| Cell-ID shuffle chance | 50.1% | 52.6% |
| Resamples above chance | 100% | 100% |
| The paper | 62.26% | 52.19% |

The engaged share matches the paper's, but disengaged events are as congruent
(81.4% of resampled engaged-minus-disengaged differences above 0), where the
paper found no bias. Whether the paper decoded the two directions together is
not stated; its Methods say "two posterior probability matrices were generated
for each event".

The 2017 tests load the 10 days and take about 3 minutes.

`test_2017_every_day_enters_by_its_decoding_error`,
`test_2017_worked_result_share_of_trajectory_events_judged_replay`,
`test_2017_congruence_as_the_methods_decode_it`,
`test_2017_congruence_decoded_jointly` and
`test_separate_decodes_cannot_tell_the_running_direction` in
`tests/datasets/test_olafsdottir_2016.py` pin these numbers;
`olafsdottir_2016.AWAKE_RESULT` and `AWAKE_CONTENT` record them.

## Cite

Ólafsdóttir HF, Carpenter F, Barry C (2016). Coordinated grid and place cell
replay during rest. Nature Neuroscience 19, 792-794.
<https://doi.org/10.1038/nn.4291>

Data: Ólafsdóttir HF, Carpenter F, Barry C (2021). LFP and single-unit data
published in Olafsdottir, Carpenter and Barry (2016) Nature Neuroscience 19:
792-794 [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.5566548>
