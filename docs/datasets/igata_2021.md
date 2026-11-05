# Igata et al. (2021): rats learn a new route and CA1 replays it

**The question.** When a rat has to change its route to a reward, does the
hippocampus replay the new route, and when? Rats ran from a start area through a
1 m open field to a goal area, collecting a small reward at a checkpoint on the
way. Partway through a day the checkpoint moved, and the rats learned the
shorter route through its new place. The dataset suits awake sharp-wave ripple
work, place maps through learning, and behavior on a free-choice lattice.

| | |
| --- | --- |
| Dataset id | `igata_2021` |
| Animal and region | Long Evans rats, dorsal CA1 |
| Apparatus | 1 m square field in a 5 x 5 lattice of 20 cm squares, with a start area, a goal area and an L-shaped return path, on a 1.4 x 1.2 m platform |
| Archive | Mendeley Data 10.17632/4xk5w69yr5, version 2 (CC-BY-4.0), 8.9 GB |
| Sessions | 40. The default `180912_detourG01` (rat 1, 114 units) is checked by hand |
| LFP | 2 kHz, every tetrode and a reference, in the session's own files |

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog().query("dataset == 'igata_2021'")
sessions[["session", "n_units", "size_bytes", "verified", "default"]].head(10)
```

<!-- docs-test: skip needs a download -->
```python
from neurospatial import compute_spatial_rates
from neurospatial.datasets import igata_2021, load

rec = load("igata_2021")              # 4.3 GB: its 124 files, read from a zip
track = igata_2021.path_track()       # the paper's paths as a 1-D environment
rates = compute_spatial_rates(
    track, rec.spike_times, rec.times, rec.positions, min_speed=5.0, bandwidth=5.0
)
```

## What was recorded

Tetrode units and LFP in dorsal CA1. A trial starts with a nose poke in the start
area and a 10 s delay. Door 1 then opens, and the rat crosses the field, visits
the checkpoint for 20 µL of chocolate milk, and reaches the goal area for 200 µL.
It returns along the outer alleyway. The field is open: a short pole stands at each
grid point, so the rat picks its own route. Position is three red LEDs on the
electrode assembly, filmed from the ceiling at 24.5 Hz.

The checkpoint is the lattice the file calls S (the paper's C1) in the `S`
sessions and G (the paper's C2) in the `G` sessions, which is how the sessions are
named. So on a learning day `<day>_detourS01` is the prelearning phase and
`<day>_detourG01` is what follows: the reward moves as it starts, and
`rec.epochs["learning"]` and `rec.epochs["postlearning"]` split it at the paper's
learning point.

The 40 sessions come in three groups:

| Group | Sessions | Units | What they are |
| --- | --- | --- | --- |
| Learning days, `Learning_data.zip` | 10: 180912 (rat 1), 180824 (rat 2), 180208 (rat 3), 180303 (rat 4), 180107 (rat 5), an `S` and a `G01` each | 114, 72, 68, 79, 22 | The paper's five rats, with its stated unit counts |
| Day after rat 1's | 2: `180913_detourG01`, `G02` | 111 | The checkpoint stays at C2, like the SI's day-2 condition. The archive does not name the rat |
| Other days | 28 | 7 on 180606, 0 on the rest | Mostly closed-loop stimulation, with pulse times in `rec.events["stimulation"]`. They offer behavior and LFP, and a watch-out says so |

The default is the session with the most units on a day the paper analysed.

## Units and conversions

| What the loader did | Basis | Evidence |
| --- | --- | --- |
| times: `log` column 0, ms / 1000 | stated | The readme: "video timestamp (ms)". |
| positions: `log` columns 1 and 2, mm / 10 | stated | The readme: "animal pos x (mm)"; the tracked span fits the SI's platform. |
| spike times: `spike` / 1000 + the file's first frame time | inferred | The readme says only "spike timestamps (ms)". The values run from 0 to the file's span, are whole ticks of 30 kHz, bin exactly into `spike_hist` from the file's start in every file of the default session, and on that clock the per-frame `cell` counts are the spikes around each frame, at no lag. |
| stitching: a return run's first frame, identical to its trial's last, is dropped | inferred | All 61 such pairs in the default session are identical rows. |
| epochs and events: runs of the `log` flags | stated | The readme names the columns. |
| checkpoint: lattice `S` or `G` from the session's name | inferred | The flag rises in that lattice on every trial of the learning days. |
| LFP: stored values, `unit="arbitrary"` | inferred | No source read gives a unit. They step by 0.25. |
| LFP times: a file's samples spread evenly over its span | inferred | Sample count / 2 kHz equals the frames' span within 1 ms in the learning days' files. |

`spike` is a pickle written by Python 2. It goes through a restricted unpickler
that lists its imports first and allows only the names a pickled float array
needs. `spike_hist` holds the same spikes in 1 ms bins; `spike` keeps the 30 kHz
times.

The sampling rate's stated value in the registry is 24.8 Hz, marked inferred: the
check reads the median frame interval. The SI's 24.5 Hz is right for the mean,
which is 24.49 to 24.50 Hz on every session; frames arrive 36 to 47 ms apart,
unevenly enough to lift the median by 1.2%.

## Watch out

- **Two sessions fail the units check.** `190910_detourG01` and
  `190920_detourG01` span 182 cm, past the 168 cm the check allows (140 cm + 20%).
  They have no units. `accept_failed_checks=("units",)` loads them.
- **No sorted units on most days.** Twenty-five sessions have none; a watch-out
  says so.
- **Stimulation.** The archive does not say whether a day's pulses followed each
  ripple, came 250 ms later, or were the nonlearning control, and they mark the
  LFP.
- **Missing files.** Some trial or return files are not in the archive, which
  leaves gaps up to 197 s in the default session. They stay gaps. A return run
  follows its trial at once, and the next trial starts one frame later, so about
  40 ms between each pair has no spikes or LFP.
- **The frame is flipped.** File y grows towards the paper's bottom edge, so a
  plot with y up shows the paper's figure upside down.
- **Lattice letters.** The paper's lower-case `u`, `e`, `s`, `g` are the file's
  `U`, `E`, `S`, `G`, and are its S, G, C1 and C2. So the paper's S is the file's
  `U`, and the file's `S` is C1. `rec.env.regions` uses the paper's names.
- **Undocumented columns.** The readme lists 14 `log` columns, and only rat 5's
  files have 14; the rest have 16, and two of those are undocumented and unread.
- **The LED flag.** Column 3 marks 3% to 29% of frames as LED undetected, yet the
  positions keep moving there. No source says what the tracker reports, and the
  positions are kept.
- **Doors.** The apparatus has automatic doors, and no file says when each was
  open. `rec.env` leaves them open.
- **Candidates are not replay.** See the worked result: most candidates here are
  not separated from chance by the detector alone.
- **No rest.** The rest-box recordings the SI mentions are not in the archive.
- **LFP files.** `rec.lfp.fetch()` reads the file's representative tetrode. In
  `190523_detourG01` and `190523_detourS01`, `lfps` has one row for five tetrode
  names and it is not `lfp`, so only the representative is offered.

## Analysis choices

`rec.params`: `bin_size` 5 cm for `rec.env`; `track_bin_size` 2 cm, the paper's;
`min_speed` 5 cm/s, the paper's threshold for stops and place fields; `dt` 0.49 s,
its 12-frame behavioral decoding window; `train_epochs` and `test_epochs`,
alternating trials; and on the learning days `learning_point`, the number of
`G01` trials before the paper's learning point.

`igata_2021.path_track()` joins the paper's straight path lines (SI Fig. S1I)
into one 1-D environment, with lines through the start area, the goal area and the
return path. The paper decodes each path on its own line; this graph holds them
on one axis so `IGATA_2021`'s place-field shuffle can run.

## Worked result

**Status: not possible yet. Nothing here is presented as replay.**

The paper defines trajectory events and plots their share, but states no share in
the text, so there was no number to match. The question was whether the paper's
candidate events, decoded and judged here, show replay. The answer on these files
is that the test cannot tell.

**The detector stops at candidates.** `ripple_detection`'s `igata_2021` finds
population-rate bursts, and has no speed rule on events. The Gaussian-mixture
split the SI applies next is under-specified. The 4,121 events it found in 1 h of
ripple-free simulated spikes were candidates too.

**The baseline was fixed in writing before any detection.** Each unit's spikes
were shifted circularly by their own offset (20 draws, seed 2026), which destroys
synchrony, and the detector was run again. Each candidate's ripple power was also
compared with 100 random stopped windows of its duration. Candidates "stand out"
if they outnumber the 95th percentile of the null counts and more than 5% beat
those windows. An event "survives" if it beats the windows and its peak rate tops
the 95th percentile of null peaks.

| Session | Stopped candidates, per stop minute | Null 95th percentile | Beat the windows | Stand out | Survivors | Judged, real | Judged, three nulls | Survivors judged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rat 1 `G01` (default) | 1,688, 46.7 | 1,986 | 66% | no | 263 | 7 of 1,688 | 1, 7, 4 | 0 |
| Rat 1 `S01` | 250, 42.3 | 252 | 64% | no | 35 | not run | not run | not run |
| Rat 2 `G01` | 1,041, 42.9 | 1,010 | 65% | yes | 93 | 9 of 1,041 | 1, 3, 7 | 0 |
| Rat 3 `G01` | 1,743, 63.5 | 1,695 | 19% | yes | 18 | 16 of 1,743 | 14, 5, 6 | 0 |
| Rat 4 `G01` | 2,090, 57.6 | 2,042 | 56% | yes | 222 | 5 of 2,090 | 15, 4, 7 | 0 |

Rat 5 has too few units for the paper's own sequence analyses, and was not run.

**What this shows.**

- About 1 event per second of stopping passes the candidate stage, on real spikes
  or shifted ones. On the default session the shifted spikes give more. Shifting
  moves running spikes into stops, which raises the stopped rate from 0.355 to
  0.444 Hz per unit and lowers the threshold from 1.171 to 1.119 Hz, so this null
  is generous to chance.
- The candidates are a mixture. Those that pass both tests look like sharp-wave
  ripples: a sharp deflection, a ripple-band burst and a population burst. Many
  others have no ripple at all.
- `IGATA_2021` judges under 1% of them, close to its share on shifted spikes.
  Rats 2 and 3 clear the fixed rule by two events each, and rat 4 does not.
- Those judged events are the shortest candidates, a median of 68 ms on the default
  session against 130 ms for all. None of the 596 survivors is judged when they are
  judged on their own (the smallest p is 0.07). One survivor is among rat 3's 16 in
  the larger run, where its surrogates draw another seed.
- The maps are not the problem: held-out running decodes 11.6 cm from the rat, in
  0.49 s bins, against 65.9 cm with shuffled positions.

The likely cause is the preset's surrogate test on these inputs: surrogates often
hold a short trajectory too, and more so in a longer event, so p rises with
duration. `IGATA_2021`'s notes warn of this where 40 cm is a large share of the
environment. Here one MAP runs over all the joined paths, where the paper took one
per path. That cause is an explanation we have not tested.

A mixture split read one way (peak rate and peak ripple power, two components each,
keeping the higher of either) keeps 1,120 of the default session's 1,688
candidates and all 263 survivors, 0.52 Hz of stopping, and 0.47 Hz on rat 1's
prelearning session, where the paper states "below 0.4 Hz". It is a diagnostic: it
decides nothing above.

**One published number does reproduce**, from behavior. By the SI's rule on the
files' lattice strings, the five rats take 25.8 +/- 9.1 trials (mean, SEM) to reach
the learning point; the paper states 26 +/- 9. The window's alignment is not stated:
a centred window is used, and a trailing one gives 27.4 +/- 8.8.

`igata_2021.WORKED_RESULT`, `igata_2021.CANDIDATE_BASELINE` and
`igata_2021.LEARNING_RESULT` hold these numbers.

## Cite

Igata H, Ikegaya Y, Sasaki T (2021). Prioritized experience replays on a
hippocampal predictive map for learning. Proceedings of the National Academy of
Sciences 118, e2011266118. <https://doi.org/10.1073/pnas.2011266118>

Data: Igata H, Ikegaya Y, Sasaki T (2020). Prioritized experience replays on a
hippocampal predictive map for learning (Version 2) [Data set]. Mendeley Data.
<https://doi.org/10.17632/4xk5w69yr5.2>
