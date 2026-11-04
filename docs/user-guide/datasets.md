# Public Datasets

You want to try an analysis on a real recording: place fields from a rat in a
maze, head-direction tuning from a mouse's thalamus, decoding from a bat in
flight. Finding the archive, reading its file layout, and working out the
units, the clock and the apparatus can take longer than the analysis.
`neurospatial.datasets` does that work once, for six published datasets, and
hands you the arrays neurospatial's functions take.

```bash
pip install "neurospatial[datasets]"
```

Importing `neurospatial.datasets` needs no extra. `load` asks for the
`datasets` extra, and says how to install it before it downloads anything.

## The datasets

| Dataset | Animal and region | Apparatus | Archive (license) | Worked result |
| --- | --- | --- | --- | --- |
| [`comrie_2026`](../datasets/comrie_2026.md) | rat, hippocampus | spatial bandit maze (track graph) | Zenodo 23020757 (CC-BY-4.0) | a sanity property, not a paper finding |
| [`grosmark_2016`](../datasets/grosmark_2016.md) | rat, dorsal CA1 | linear and circular mazes, with sleep before and after | OSF `hu5ma` (license not stated); DANDI 000044 (CC-BY-4.0) | reproduced (level B) |
| [`peyrache_2015`](../datasets/peyrache_2015.md) | mouse, anterior thalamus | open arena, head direction only | OSF `jb2gd` (license not stated) | reproduced (level A) |
| [`sargolini_2006`](../datasets/sargolini_2006.md) | rat, medial entorhinal cortex | 1 m open field | DANDI 000582 (CC-BY-4.0) | not possible yet |
| [`forli_2025`](../datasets/forli_2025.md) | Egyptian fruit bat, dorsal hippocampus | flight room, 3-D | Zenodo 15738988 (CC-BY-4.0) | not possible yet; a 3-D attempt did not reproduce the paper (15.2% of units tuned against 71%) |
| [`valero_2022`](../datasets/valero_2022.md) | mouse, CA1 | linear track, with video | DANDI 000568 (CC-BY-4.0) | not possible yet |

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
package pins, and keeps it in the cache; later calls read the cache. Before it
returns, the loader checks the session:

- **Refusing checks.** The *clock* check refuses a session whose spikes,
  events or epochs fall outside the tracked span (or an untracked epoch such as
  sleep). The *units* check refuses a session whose largest position span or
  sampling rate disagrees with what a source states. Both raise
  `DatasetCheckError`, which gives the numbers and the argument that loads the
  session anyway.
- **Warning checks.** Samples at exactly the origin, runs of identical
  positions, samples outside the environment, and units with no spikes in the
  analysed epochs are described in "Watch out" with their numbers. No array is
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
longest gap, speed, head-direction coverage), **Environment** (how it was
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
environment with the path, occupancy (never-visited bins drawn as missing), the
speed histogram with `params["min_speed"]` marked, head direction beside
heading minus direction of travel, units per brain region, and the first 60 s
of position beside the spike raster. A panel the dataset has no data for says
"not in this dataset".

<!-- docs-test: run setup=docs_datasets_session -->
```python
fig = rec.plot_overview()
fig.savefig("overview.png", dpi=100)
```

For the time course, `rec.animation()` builds an `Animation` of the session:
occupancy over the environment, with the path, headings, spike raster, events
and epochs of the window on top. It starts with the first 60 s of tracking,
so the first view opens fast; pass `time_range=(start, stop)` in seconds for
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

Of the six datasets, only [`valero_2022`](../datasets/valero_2022.md) has
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

**What the overlay shows, and what it does not.** The Valero files state no
pixel size and store no pixel tracking, so the loader calibrates the video by
fitting the LED it detects in the frames to the file's positions. The overlay
is therefore aligned with the tracking by construction: it shows what the
camera saw at each moment, not a check of the tracking or of the declared
track. `rec.video.calibration_basis` gives the fit and its residual, and a
watch-out repeats the caveat. A session without a calibration refuses
`animation(video=True)` with an `InputError` that names each source tried; the
frames are never stretched over the environment's extent.

## Move to pynapple or NWB

The arrays hand off to other tools unchanged. To pynapple, `to_pynapple`
turns timestamps and values into a `Tsd` or `TsdFrame`; the spike trains and
epochs map directly onto pynapple's own classes:

<!-- docs-test: skip needs the pynapple extra -->
```python
import pynapple as nap

from neurospatial.io import to_pynapple

position = to_pynapple(rec.times, rec.positions)          # TsdFrame, columns x, y
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

A file is downloaded once. Before it starts, `load` logs a warning (shown by
default) with the download's size and URL, and the summary's `Source` line
gives the file's size and URL afterwards. A download whose checksum differs
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
list of paths). It logs the total size first, as a warning, and refuses to
start if the cache's disk has less free space than that. For `grosmark_2016`
that total is about 66 GB.

## CRCNS credentials

None of the six datasets needs an account. The package's download helpers
for files at crcns.org read the login from the `CRCNS_USERNAME` and
`CRCNS_PASSWORD` environment variables and never print them or put them in an
error message; set both in your shell before using them, not in code.

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
for dataset, session in zip(small["dataset"], small["session"]):
    rec = load(dataset, session)
    rows.append({
        "dataset": dataset,
        "session": session,
        "species": rec.animal,
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
