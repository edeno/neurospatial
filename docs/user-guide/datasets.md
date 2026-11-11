# Public Datasets

You want to try an analysis on a real recording: place fields from a rat in a
maze, head-direction tuning from a mouse's thalamus, decoding from a bat in
flight. Finding the archive, reading its file layout, and working out the
units, the clock and the apparatus can take longer than the analysis.
`neurospatial.datasets` does that work once, for seventeen published datasets, and
hands you the arrays neurospatial's functions take.

```bash
pip install "neurospatial[datasets]"
```

Importing `neurospatial.datasets` needs no extra. `load` asks for the
`datasets` extra, and says how to install it before it downloads anything.

## The datasets

The last column says how far each dataset's result was checked.

- **Reproduced:** we matched a number or finding from the paper.
- **Tested against chance:** we ran the analysis with a pass mark fixed in advance,
  and compared the result with the same analysis on data whose structure was broken
  on purpose. This is not a finding of the paper. *Not met* means the result did not
  clear the mark, and the dataset's page says what that can and cannot mean.
- **Not possible yet:** the paper's result cannot be checked yet, and the page says why.

| Dataset | Animal and region | Apparatus | Archive (license) | Worked result |
| --- | --- | --- | --- | --- |
| [`comrie_2026`](../datasets/comrie_2026.md) | rat, hippocampus | spatial bandit maze (track graph) | Zenodo 23020757 (CC-BY-4.0) | tested against chance, not a paper finding |
| [`grosmark_2016`](../datasets/grosmark_2016.md) | rat, dorsal CA1 | linear and circular mazes, with sleep before and after | OSF `hu5ma` (license not stated); DANDI 000044 (CC-BY-4.0) | reproduced (level B) |
| [`peyrache_2015`](../datasets/peyrache_2015.md) | mouse, anterior thalamus | open arena, head direction only | OSF `jb2gd` (license not stated) | reproduced (level A) |
| [`sargolini_2006`](../datasets/sargolini_2006.md) | rat, medial entorhinal cortex | 1 m open field | DANDI 000582 (CC-BY-4.0) | not possible yet |
| [`forli_2025`](../datasets/forli_2025.md) | Egyptian fruit bat, dorsal hippocampus | flight room, 3-D | Zenodo 15738988 (CC-BY-4.0) | not possible yet; a 3-D attempt did not reproduce the paper (15.2% of flight-active units tuned against 71%), and our own 3-D decode of its ripple events failed its sanity check, with 0.3% of planted replay judged |
| [`valero_2022`](../datasets/valero_2022.md) | mouse, CA1 | linear track, with video | DANDI 000568 (CC-BY-4.0) | not possible yet |
| [`vollan_2025`](../datasets/vollan_2025.md) | rat, medial entorhinal cortex and hippocampus | open field (box or circle) and 200 cm linear track, head direction with position | EBRAINS 10.25493/R5FR-EDG (CC-BY-SA-4.0) | not possible yet; module grid scores match the authors' for modules 1 and 2, not module 3 (0.185 against 0.685) |
| [`gardner_2022`](../datasets/gardner_2022.md) | rat, medial entorhinal cortex grid modules | 1.5 m open field, head direction with position | Figshare 16764508 (CC0) | not possible yet |
| [`peyrache_rsc`](../datasets/peyrache_rsc.md) | mouse, retrosplenial cortex | square and 70 cm circular arenas, head direction with position, sleep | DANDI 000987 (CC-BY-4.0) | not possible yet |
| [`kanter_2025`](../datasets/kanter_2025.md) | rat, lateral and medial entorhinal cortex, CA1 or CA3 | 1 to 2 m open field, empty or with novel objects | EBRAINS 10.25493/GTVB-FQR (CC-BY-SA-4.0) | not possible yet |
| [`shipley_2026`](../datasets/shipley_2026.md) | mouse (App NL-G-F and wild type), CA1 | 8-arm radial maze (track graph), with rest | UCL RDR 30902972 (CC-BY-4.0) | not possible yet |
| [`huszar_2026`](../datasets/huszar_2026.md) | mouse, CA1 and CA3 | 90 cm cheeseboard maze, learning trials, reward wells, ripples | DANDI 001880 (CC0-1.0) | not possible yet |
| [`grieves_2020`](../datasets/grieves_2020.md) | rat, CA1 | 3-D climbing lattice (0.97 m cube, aligned or tilted) | UCL RDR 9926477 (CC0) | not possible yet |
| [`igata_2021`](../datasets/igata_2021.md) | rat, dorsal CA1 | 1 m lattice field with a start, a goal and a return path; the rat learns a new route; LFP | Mendeley Data 4xk5w69yr5 (CC-BY-4.0) | not possible yet; nothing is presented as replay: on the default session the candidate events do not stand out from a shifted-spike baseline, and `IGATA_2021` judges none of the 596 events that pass it over four rats' sessions, which says little since it judges 0.7% of planted replay. Trials to the learning point match the paper (25.8 +/- 9.1 against 26 +/- 9) |
| [`bhattarai_2020`](../datasets/bhattarai_2020.md) | rat, dorsal CA1 | loop with two rings, four sequences to one reward; one LFP channel | Figshare 10032866 (CC0) | not possible yet; the paper's replay recipe on the archive's one LFP channel passes 37 candidates against the paper's 278 on the default set, and 11 judged replay (29.7%) only just clears its sanity check (24.6% at the 95th percentile with permuted cells); planted replay is judged 18.6% of the time, so this is no rate of replay; the complex-spike rule gives the paper's 648 of 698 units |
| [`huszar_2022`](../datasets/huszar_2022.md) | mouse, CA1 | figure-eight maze with sleep; LFP for 12 sessions | DANDI 000552 (CC-BY-4.0) | tested against chance of our own analysis, not the paper's: 5 of 20 awake ripple candidates are judged trajectories, none in 40 shifted-spike draws (20 candidates give no rate, a null of zeros cannot show calibration, and this judge passes 8.7% of planted replay) |
| [`carey_2019`](../datasets/carey_2019.md) | rat, dorsal CA1 | T-maze with food and water arms, rest pedestals; LFP per channel | DataLad MotivationalT (ODC-PDDL-1.0) | tested against chance, **not met**: on the route of the paper's code, 41 sequences on 1,328 candidates against a null 95th percentile of 52.35, with 5.8% of planted replay found, so it cannot show whether these events hold replay. Detection matches: our score matches the paper's stored score (r = 0.9999997); 73.5% and 91.6% agreement with its 1,654 events, 98.6% and 97.2% with its speed and theta |
| [`shin_2019`](../datasets/shin_2019.md) | rat, dorsal CA1 and prefrontal cortex | W-maze learned in one day, 8 run sessions between rest; LFP of every tetrode | DANDI 000978 (CC-BY-4.0), read by byte range | not possible yet; the paper's per-visit counts on its six rats meet five of six marks fixed before the run: 9.46 s of immobility, 6.68 SWRs, 2.98 candidates and 1.02 replay events per visit (paper 10.3, 7.3, 2.3, 0.9), but 75.5% of visits with two or more SWRs against 85.7%. The beam times, interneuron labels and ripple tetrodes the paper used are not in the files |

Each dataset has a page with what was recorded, the conversions the loader
applied and the evidence for them, the pitfalls, the worked result and how to
cite it. A "worked result" is a finding of the paper that the page reproduces
with public neurospatial functions, at a stated level; "not possible yet"
says what is missing instead.

## See what is available

`catalog()` lists every session of every dataset without downloading
anything: one row per session.

<!-- docs-test: run -->
```python
from neurospatial.datasets import catalog

sessions = catalog()
print(sessions[["dataset", "session", "species", "n_units", "verified"]].head())

# The sessions a person has checked by hand, one default per dataset.
print(sessions.query("verified")[["dataset", "session", "n_units", "default"]])
```

The columns are `dataset`, `session`, `species`, `brain_regions`, `apparatus`,
`n_units` (missing where nobody has opened the file), `verified`, `default`,
`license`, `archive`, `size_bytes` (what loading the session downloads) and
`citation`. `license` is what the archive states for the file; "not stated"
means the archive states none, so check with the data's owners before you
redistribute it.

`verified` means a person has checked the session by hand against its archive,
its files and the sources its page lists (not always the paper: each
dataset's page says which sources were read). Every dataset has one default session, which `load` gives when
you name none. Where one exists it is the session the tutorials of other tools
use. The other sessions load too, with the same automatic checks, but their
summary opens with "Not yet checked by hand".

## Load a session

<!-- docs-test: skip needs a download -->
```python
from neurospatial.datasets import load

rec = load("sargolini_2006")                      # the default session
rec = load("sargolini_2006", "sub-10884_ses-03080402")   # a named one
```

The first call downloads the file, checks its SHA-256 against the one the
package pins, and keeps it in the cache; later calls read the cache, hashing
the file again only if its size or modification time changed. Before it
returns, the loader checks the session:

- **Refusing checks.** The *clock* check refuses a session whose spikes fall
  outside the tracked span (or an untracked epoch such as sleep), or whose
  events or epochs fall outside the whole recording, from its first tracking
  sample or spike to its last (an event during sleep is on the clock). The *units* check refuses a session whose largest position span or
  sampling rate disagrees with what a source states. Both raise
  `DatasetCheckError`, which gives the numbers and the argument that loads the
  session anyway.
- **Warning checks.** Samples at exactly the origin, runs of identical
  positions, samples farther from the last believable sample than the
  dataset's stated speed limit allows (where none is stated, speed samples
  more than twice the 99.9th percentile of speed: likely tracking jumps, a
  heuristic), samples outside the environment, and units with no spikes in
  the analysed epochs are described in "Watch out" with their numbers. Where a
  track's width was set from the samples themselves
  (`rec.track_width_quantile`), the share outside holds by construction (about
  1 % at the 99th percentile), and the finding says so. No array is
  changed: positions stay as the file stores them, and an untracked sample
  stays NaN.

To load a session that fails a refusing check, name the check:

<!-- docs-test: skip needs a download -->
```python
rec = load("valero_2022", "sub-fCamk2_ses-fCamk2-201013-sess2",
           accept_failed_checks=("clock",))
print(rec.failed_checks)        # ('clock',), and the first "Watch out" line
```

Only `"clock"` and `"units"` can be named. Naming a check that passes changes
nothing. The accepted failure travels with the recording in
`rec.failed_checks` and heads "Watch out", so it is in front of whoever reads
the summary next.

## Read the summary

Printing a recording shows what it is and what to watch for. A recording from
a short simulated session shows the layout:

<!-- docs-test: run setup=docs_datasets_session -->
```python
print(rec)
```

The sections are, in order: the identity lines (paper, animal, session,
apparatus, archive, license, citation), **Time** (duration, clock, sampling
rate), **Epochs**, **Tracking** (what was tracked, the tracked share, the
longest gap, speed: median, 95th percentile and fastest step, head-direction
coverage), **Environment** (how it was
built, and from which source), **Neurons** (counts per brain region and the
unit labels the authors gave), and **Watch out**, always last and always in
full.

"Watch out" holds the pitfalls a person found when checking the session, and
the ones the automatic checks measured. Read it before you analyse: it says,
for example, which units are not the ones a tutorial uses, that a position is
a projection rather than a 2-D track, or that a heading frame is assumed. Each
dataset page repeats the pitfalls and gives their evidence.

`rec.summary()` returns the same numbers as a flat dictionary (`n_units`,
`duration`, `n_dims`, `sampling_rate`, `tracked_fraction`, `longest_gap`,
`median_speed`, `speed_p95`, `verified`), and each key is also an attribute of
`rec`.

Every conversion the loader applied is in `rec.conversions`, a tuple of
`Conversion(text, basis, evidence)`. A `basis` of `"stated"` means a source
the page names says it; `"inferred"` means the loader worked it out, and
`evidence` gives the numbers. `rec.provenance` says where the file came from
(archive, URL, SHA-256, size, neurospatial version, time fetched).

## Use the arrays

The attributes are what the analysis functions take, in the package's
argument order:

<!-- docs-test: run setup=docs_datasets_session -->
```python
from neurospatial import compute_spatial_rates

rates = compute_spatial_rates(rec.env, rec.spike_times, rec.times, rec.positions)
print(rec.n_units, rec.duration, rec.position_units, rec.env.n_bins)
```

`rec.params` holds the analysis choices the dataset's worked result used
(`bin_size`, `min_speed`, `dt`, and `train_epochs` and `test_epochs` for
held-out decoding). They are defaults to start from, each justified on the
dataset's page, not requirements. Positions, headings and times are read-only
arrays; `rec.positions`, `rec.headings`, `rec.linearized` and `rec.env` are
`None` where the dataset has none (`peyrache_2015` has headings but no
positions).

## Look at it

`plot_overview` draws six panels that show the session at a glance: the
environment with the path (on a 1-D track, a histogram of the samples along
it; in 3-D, a top view, as is occupancy, summed over z), occupancy (never-visited bins drawn as missing), the
speed histogram up to the 99.9th percentile with `params["min_speed"]` marked
(the panel counts the faster samples and gives the fastest), head direction beside
heading minus direction of travel, units per brain region, and the first 60 s
of position beside the spike raster, from the later of the first tracked
sample and the first spike, so a session whose neural recording starts after
its tracking still shows spikes, and from the first analysed epoch after that
where `params["analysis_epochs"]` names them (the bat's flights). For
hippocampal units on a 1-D track the raster is sorted by each unit's
place-field peak along the track, so sequences read as diagonals. A panel the dataset has no data for says
"not in this dataset".

<!-- docs-test: run setup=docs_datasets_session -->
```python
fig = rec.plot_overview()
fig.savefig("overview.png", dpi=100)
```

For the time course, `rec.animation()` builds an `Animation` of the session:
occupancy over the environment, with the path, headings, spike raster, events
and epochs of the window on top. It starts with the same 60 s as the
overview's time panel, so the first view opens fast and shows spikes; pass `time_range=(start, stop)` in seconds for
another window.

<!-- docs-test: skip needs a GUI window -->
```python
scene = rec.animation()
scene.show(backend="fastplotlib")       # an interactive GPU viewer
```

`show(backend="fastplotlib")` needs the `fastplotlib` extra (Python 3.11 or
newer). To share a clip instead, export a window; a 10 s export at 10 frames
per second is 100 frames:

<!-- docs-test: run setup=docs_datasets_session -->
```python
start = rec.times[0]
scene = rec.animation(time_range=(start, start + 10.0))
scene.export("session.mp4", fps=10)
```

Video files are encoded with the ffmpeg that the `video` extra installs (see
[Field Animation](animation.md)). A session without positions
(`peyrache_2015`) raises `InputError`; use its `headings` and `spike_times`
directly.

## Draw the data over the camera video

Where the dataset publishes the session's video, `rec.video` is a
`VideoSource`: the file, one time per frame, the pixel-to-cm calibration, and
the basis of each. It is `None` where there is no video. Nothing is downloaded
until you ask, by `rec.video.fetch()` or by `rec.animation(video=True)`; the
file is verified against its SHA-256 and kept beside the session's file in the
cache. It needs the `video` extra (OpenCV).

Of the fourteen datasets, only [`valero_2022`](../datasets/valero_2022.md) has
video:

!!! warning "The video overlay is aligned with the tracking by construction"
    The Valero files state no pixel size and store no pixel tracking, so the
    loader calibrates the video by fitting the LED it detects in the frames to
    the file's positions. The overlay shows what the camera saw at each moment.
    It is not a check of the tracking or of the declared track.

<!-- docs-test: skip needs a download -->
```python
rec = load("valero_2022")
start = rec.times[0]
scene = rec.animation(time_range=(start, start + 20), video=True)
scene.export("valero_video.mp4", fps=30)
```

`rec.video.calibration_basis` gives the fit and its residual, and a watch-out
repeats the caveat above. A session without a calibration refuses
`animation(video=True)` with an `InputError` that names each source tried; the
frames are never stretched over the environment's extent.

## Read the LFP

Where the dataset publishes the local field potential, `rec.lfp` is an
`LFPSource`: the files that hold it, the sampling rate, what each channel is, the
unit and how the times reach the recording's clock. It is `None` where there
is no LFP. LFP is often much larger than the rest of a session, so nothing is
downloaded at load; the summary says it exists and how large it is.
`rec.lfp.fetch()` downloads what the channels you ask for need, verifies each
file against its SHA-256, and returns the samples and their times. With no
channels it reads `rec.lfp.ripple_channels` where the dataset states them, and
every channel otherwise, up to 32 channels. A source of more channels with no
stated ripple channel raises an error that gives its size, because its files
can run to hundreds of gigabytes and its samples to more than fit in memory:
choose from `rec.lfp.channels`. `tuple(rec.lfp.channels)` reads them all:

<!-- docs-test: skip needs a download -->
```python
rec = load("forli_2025")
print(dict(rec.lfp.channels))  # each channel, and what it is
samples, lfp_times = rec.lfp.fetch(rec.lfp.ripple_channels)
```

`samples` has shape `(n_time, n_channels)`, in `rec.lfp.unit`, with columns in
the order you asked for. `lfp_times` are seconds on the same clock as
`rec.spike_times`, and are not on the tracking samples of `rec.times`.
`rec.lfp.ripple_channels` names the channels the paper chose for ripple
detection, where it states them. `rec.lfp.unit_basis` and
`rec.lfp.times_basis` say where the scale and the times come from, as
`rec.conversions` does for the rest of the session.

Ask for the channels you need: [`forli_2025`](../datasets/forli_2025.md) holds 384
channels, one 53 MB in memory and all of them 405 GB, and
[`huszar_2022`](../datasets/huszar_2022.md)'s LFP sits in raw files of up to
184 GB, which `stream=True` reads by byte range. Five datasets have an LFP:
`forli_2025`, `igata_2021` (every tetrode of the session, read from its own
files), `bhattarai_2020` (one channel), `carey_2019` (each channel from its own
file) and `huszar_2022` (12 sessions). [`shin_2019`](../datasets/shin_2019.md)
holds the LFP of every tetrode inside its 28 to 47 GB files, which are never
downloaded: `rec.lfp.fetch(channels, stream=True)` reads 14 to 24 GB however
few channels you ask for, so ask for all you need in one call.

## Move to pynapple or NWB

The arrays hand off to other tools unchanged. To pynapple, `to_pynapple`
turns timestamps and values into a `Tsd` or `TsdFrame`; the spike trains and
epochs map directly onto pynapple's own classes:

<!-- docs-test: skip needs the pynapple extra -->
```python
import pynapple as nap

from neurospatial.io import to_pynapple

position = to_pynapple(rec.times, rec.positions, columns=["x", "y"])  # TsdFrame
heading = to_pynapple(rec.times, rec.headings)            # Tsd, radians
spikes = nap.TsGroup(
    {int(u): nap.Ts(t) for u, t in zip(rec.units.index, rec.spike_times, strict=True)}
)
```

To NWB, `write_environment` stores the recording's environment (bins, regions,
landmarks) in the file's `/analysis` group, so a later session reads the same
bins back with `read_environment`:

<!-- docs-test: skip needs the nwb extra -->
```python
from datetime import datetime, timezone

from pynwb import NWBFile, NWBHDF5IO

from neurospatial.io.nwb import write_environment

nwbfile = NWBFile("Session", "session", datetime(2026, 1, 1, tzinfo=timezone.utc))
write_environment(nwbfile, rec.env, name=rec.session)
with NWBHDF5IO("session.nwb", "w") as io:
    io.write(nwbfile)
```

See [Interoperability](interoperability.md) for reading positions, units and
epochs from NWB files. A session whose environment is `None` has nothing to
write.

## The cache, and downloading versus streaming

Files are cached as `<cache>/<dataset>/<file name>`, or, for a session read
from members of a remote zip, as `<cache>/<dataset>/<session>/<member name>`,
where `<cache>` is the
first of the `cache=` argument, the `NEUROSPATIAL_DATA_DIR` environment
variable, and the per-user default: `~/.cache/neurospatial/datasets`
(`%LOCALAPPDATA%\neurospatial\datasets` on Windows). Set the variable once to
keep the data on a disk with room:

```bash
export NEUROSPATIAL_DATA_DIR=/data/neurospatial
```

A file is downloaded once. Before it starts, `load` issues a `DownloadWarning`
(shown by default, pointing at your `load` call) with the download's size, URL
and cache directory, and the summary's `Source` line gives the file's size and
URL afterwards. Once you expect the downloads,
`warnings.simplefilter("ignore", DownloadWarning)` (with `DownloadWarning`
from `neurospatial.datasets`) silences the notice. A download whose checksum differs
from the pinned one is refused and removed. A cached file whose checksum
differs is never overwritten: loading raises `FileExistsError` and asks you to
move it aside.

Sizes differ a great deal. `catalog()["size_bytes"]` is what loading the
session downloads: from 3.7 MB (a Sargolini session) to 9.2 GB (a
Grosmark DANDI session, most of it LFP). Two ways to keep it small:

- `stream=True` reads a DANDI file by HTTP byte range instead of downloading
  it, after checking that the archive still publishes the pinned SHA-256.
  Nothing is cached. Use it for the large DANDI files, for example
  `load("grosmark_2016", "sub-Achilles_ses-Achilles-10252013", stream=True)`.
  A file outside DANDI (an OSF or Zenodo file) cannot be streamed and raises
  `InputError`.
- Some sessions fetch only part of an archive: the bat's 3.3 GB Zenodo zip is
  read by byte range for the five members the loader needs (260 MB).

`fetch_all(dataset)` downloads every session of a dataset (it returns the
list of paths). Before calling it, add up what it will download:
`catalog().query("dataset == 'grosmark_2016'")["size_bytes"].sum()` is about
66 GB. It gives the total size first, as a `DownloadWarning`, and refuses to
start if the cache's disk has less free space than that.

## CRCNS credentials

None of the datasets needs an account. The package's download helpers
for files at crcns.org are internal, for loaders of CRCNS datasets to come;
they are not part of the public API. They read the login from the
`CRCNS_USERNAME` and `CRCNS_PASSWORD` environment variables and never print
them or put them in an error message; set both in your shell, not in code.

## Compare sessions across datasets

Because every session is a `Recording`, a loop over `catalog()` is a
meta-analysis. Choose sessions by what the catalog says (checked by hand,
size, species), and keep what each result needs to be interpreted:

<!-- docs-test: skip needs a download -->
```python
import pandas as pd

from neurospatial.datasets import catalog, load

small = catalog().query("verified and size_bytes < 500e6 and n_units > 0")
rows = []
for row in small.itertuples():
    rec = load(row.dataset, row.session)
    rows.append({
        "dataset": row.dataset,
        "session": row.session,
        "species": row.species,
        "n_units": rec.n_units,
        "duration_s": rec.duration,
        "position_units": rec.position_units,
        "median_speed": rec.median_speed,
        "watch_outs": len(rec.watch_outs),
    })
print(pd.DataFrame(rows))
```

Three things to keep in view when you pool sessions:

- **Units differ.** `position_units` is `"cm"`, `"m"`, `"pixel"` or
  `"arbitrary"`; the bat is in metres and the others in centimetres. Compare
  speeds and field sizes only after converting.
- **Time differs.** The clocks, sampling rates and the unit labels the
  authors gave (`rec.units["label"]`) are not shared across datasets.
- **A default is a choice.** Each dataset's `params` are not tuned across
  datasets; if you pool a result, fix its parameters yourself and record
  them.

Cite each dataset you use: `rec.citation` holds the paper's citation, then
the archive's citation for the data.
