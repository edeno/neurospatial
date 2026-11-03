# Real-data survey

Scripts that load public spatial-neuroscience recordings into plain arrays, run
a fixed set of neurospatial operations on each, and record what worked, what
needed a workaround, what neurospatial's warnings said, and where neurospatial
disagrees with another tool. The ranked proposals are in
[FINDINGS.md](FINDINGS.md), the evidence in [GAPS.md](GAPS.md), and the
tables in [results/](results/). This folder is not part of the installed
package and is not run by the test suite or CI.

Each `load_<dataset>.py` also works on its own as an example of bringing a
recording into neurospatial: it fetches the files, reads them (with
neurospatial's NWB readers where they work), converts units and frames, and
explains every analysis choice in its docstring.

## Run it

```bash
uv sync --group datasets --extra nwb --extra pynapple
MPLBACKEND=Agg uv run --group datasets --extra nwb --extra pynapple python datasets/run.py
MPLBACKEND=Agg uv run --group datasets --extra nwb --extra pynapple python datasets/reference_checks.py
```

`run.py --datasets KEY ... --ops OP_ID ...` runs a subset; `run.py --render-only`
rebuilds the Markdown tables from the JSON results. A subset run keeps the
other cells of the stored record, so it is refused when the load differs
from the stored one: its `conversions`, `env_kind` or `load_fingerprint`
(hashes of `params` and the geometry's stated entries, the positions' shape
and NaN count, the tracked span, each unit's spike count and the unit ids).
Records written before the fingerprint have none; each needs one full rerun
(without `--ops`) before `--ops` works on it. One loader alone:
`uv run --group datasets --extra nwb python datasets/load_sargolini_000582.py`.

Offline tests (simulated data, no network):

```bash
uv run --group datasets --extra nwb --extra pynapple pytest datasets/tests -n 0
```

## Downloads

Files are downloaded once into `data/datasets/` (git-ignored) or into
`$NEUROSPATIAL_DATA_DIR` when it is set, and every file is checked against a
pinned SHA-256. A file whose checksum does not match is never overwritten. Data
are not redistributed here; each loader records the dataset's license.
Zip members are read by HTTP byte range; Deflate64 members (Grieves 2020),
which Python's `zipfile` cannot inflate, are inflated with `inflate64` and
CRC-checked. The labyrinth's trajectories and Gardner's spike trains are Python pickles;
their loaders read them with restricted unpicklers that resolve only the
globals and dtypes those files hold, and refuse every other one (Gardner's
loader also lists a pickle's imports from its opcodes before unpickling).

Multi-GB DANDI files (000044, 000447, 000003) are not downloaded: the loader checks the
SHA-256 DANDI publishes for the asset, then reads the few datasets it needs by
HTTP byte range (`_fetch.open_nwb`), about a minute per file.

| Dataset | What it is | Size | License |
| --- | --- | --- | --- |
| `sargolini_000582` | Rat MEC grid/conjunctive cells, 1 m box (DANDI 000582) | 15.9 MB | CC-BY-4.0 |
| `peyrache_mouse32` | Mouse ADn head-direction cells, wake and sleep (OSF jb2gd) | 36.6 MB | unstated on OSF; source DANDI 000056 is CC-BY-4.0 |
| `figure8_000552` | Mouse CA1 on a figure-8 maze, ripples and sleep states (DANDI 000552) | 96.0 MB | CC-BY-4.0 |
| `rosenberg_labyrinth` | Mouse exploring a binary-tree labyrinth, behavior only (Rosenberg et al. 2021 repository) | 6.3 MB | MIT |
| `yartsev_bat_demo` | Bat CA1 during 3-D flight (Zenodo 15738988; members of a 3.3 GB zip read by byte range) | 260 MB | CC-BY-4.0 |
| `miniscope_002030` | Mouse CA1 one-photon calcium imaging (ΔF/F), circular open field (DANDI 002030) | 4.9 MB | CC-BY-4.0 |
| `giocomo_000405` | Rat open field with binned firing rates and mislabelled tracking (DANDI 000405, draft) | 27.1 MB | unstated (draft) |
| `giocomo_001701` | Mouse MEC and CA1 units on an X maze, head direction labelled degrees (DANDI 001701) | 13.0 MB | CC-BY-4.0 |
| `gardner_2022_grid` | Rat MEC grid modules 1–3 in an open field, 483 units (Figshare 16764508; one zip member by byte range) | 168.9 MB | CC0 |
| `vollan_lt` | Rat MEC Neuropixels units on a linear track with tracked head direction (EBRAINS 10.25493/R5FR-EDG) | 110.6 MB | CC-BY-SA-4.0 |
| `vollan_of` | Rat MEC Neuropixels grid modules in an open field, with the authors' grid scores (EBRAINS 10.25493/R5FR-EDG) | 205.7 MB | CC-BY-SA-4.0 |
| `grosmark_000044_circular` | Rat CA1 on a 1 m circular track with sleep before and after (DANDI 000044) | streamed from 9.2 GB | CC-BY-4.0 |
| `grosmark_000044_linear` | Rat CA1 on a 1.6 m linear track with sleep before and after (DANDI 000044) | streamed from 8.7 GB | CC-BY-4.0 |
| `jadhav_000447` | Rat CA1 and PFC units on the novel W-track, runs 1 and 3 (DANDI 000447) | streamed from 5.1 GB | CC-BY-4.0 |
| `jadhav_000447_familiar` | The same session on the familiar W-track, run 2 (DANDI 000447) | streamed from 5.1 GB | CC-BY-4.0 |
| `hc11_achilles` | Rat CA1 on a 1.6 m linear track, sleep before and after; the session of `grosmark_000044_linear` (CRCNS hc-11) | 38.5 MB from a 277.5 MB archive | CRCNS terms |
| `hc3_ec013_669` | Rat CA1 and entorhinal units on a 250 cm linear track, two LEDs (CRCNS hc-3) | 5.5 MB from a 233.8 MB archive, plus a 1.1 MB cell table | CRCNS terms |
| `hc2_ec013_527` | Rat CA1 units in a 180 cm open field, two LEDs (CRCNS hc-2) | 6 MB from a 211.8 MB archive | CRCNS terms |
| `hc28_hpa` | Rat CA1 and PFC units on a W-track, untracked frames stored as 0 (CRCNS hc-28) | 21.3 MB from a 2.8 GB archive | CRCNS terms |
| `hc31_anm190422` | Rat CA1 units on a 48 m track, position in inches (CRCNS hc-31) | 166.8 MB | CRCNS terms |
| `hc18_train261` | Rat CA1 units while carried on a model train (CRCNS hc-18) | 134.8 MB from a 6.5 GB archive | CRCNS terms |
| `grieves_2020_lattice` | Rat CA1 in a 3-D climbing lattice (UCL RDR 9926477; one Deflate64 member of an 11.2 GB zip by byte range) | 431 MB | CC0 |
| `watermaze_10248158` | Mouse Morris water-maze swim paths, EthoVision JSON, behavior only (Figshare 10248158) | 9.9 MB | CC-BY-4.0 |
| `olafsdottir_5566548` | Rat hippocampus and deep MEC tetrodes on a Z-track, Axona files (Zenodo 5566548; 34 members of a 9.9 GB zip by byte range) | 105 MB | CC-BY-4.0 |
| `takigawa_7114733` | Rat CA1 on two novel 2 m linear tracks (Zenodo 7114733) | 20.0 MB | CC0 |
| `hafting_2008_nird` | Rat MEC grid cells on a 320 cm linear track (NIRD 10.11582/2017.00020; 4 members of a 20.9 MB zip by byte range) | 1.0 MB | CC-BY-4.0 |
| `j16_23020757` | Rat hippocampus in a spatial bandit maze with its track graph, `.npz` without pickles (Zenodo 23020757) | 75.1 MB | CC-BY-4.0 |
| `agarwal_24757638` | Rat CA1 and mPFC on a 3-arm maze with a wheel delay (Figshare 24757638) | 226 MB | CC-BY-4.0 |
| `shipley_30902972` | Mouse CA1 tetrodes on an 8-arm radial maze and sleep, Kilosort NPY + MATLAB (UCL RDR 30902972; 4 members of a 4.4 GB zip by byte range) | 67 MB | CC-BY-4.0 |
| `rsc_000987` | Mouse retrosplenial units in a square arena, motion-capture head direction (DANDI 000987) | 20.2 MB | CC-BY-4.0 |
| `ca1ca3_001880` | Mouse CA1 and CA3 units on a cheeseboard maze, ripples (DANDI 001880) | 65.7 MB | CC0-1.0 |
| `gonzalez_001695` | Mouse CA1/CA3/RSC units on a novel linear maze with sleep, ripples (DANDI 001695) | 61.3 MB | CC-BY-4.0 |
| `neurolab_001754` | Rat CA1 units in spaceflight, Escher staircase; pixels sampled at spike times (DANDI 001754) | 3.2 MB | CC-BY-4.0 |
| `valero_000568` | Mouse CA1 units on a 110 cm linear track, ripples (DANDI 000568) | 5.8 MB | CC-BY-4.0 |
| `aeryjones_000165` | Mouse hippocampal multi-unit activity per channel on a linear track (DANDI 000165) | 23.3 MB | CC-BY-4.0 |
| `dyads_001845` | Rat on a cooperative W-maze, DeepLabCut pose, behavior only (DANDI 001845) | 1.4 MB | CC0-1.0 |
| `senzai_000003` | Mouse dentate gyrus and CA3 units, open exploration and maze; pixels labelled metres (DANDI 000003) | streamed from 4.7 GB | CC-BY-4.0 |
| `kanter_2025` | Rat LEC and CA1 Neuropixels units, 1 m open field with novel objects (EBRAINS 10.25493/GTVB-FQR) | 29.6 MB | CC-BY-SA-4.0 |

NWB loaders set `params["nwb_source"]` to the cached file, or to the URL of a
file read by byte range. The `io.nwb_readers`
operation then runs neurospatial's position, head-direction and unit readers
on it with default arguments and compares what they return with the loader's
range-checked arrays. `io.read_pose` runs `read_pose` the same way, reports each
keypoint's share of NaN samples, and feeds one keypoint to the rate maps (or
occupancy, for a behavior-only file).

CRCNS loaders (`hc*`) need a free crcns.org account. They read it from the
`CRCNS_USERNAME` and `CRCNS_PASSWORD` environment variables and never print,
log or store either value (`_fetch.fetch_crcns`). CRCNS serves `.tar.gz`
archives, which cannot be read by byte range: `_fetch.fetch_crcns_tar_members`
streams an archive once and keeps only the pinned members, so the first load
of `hc28_hpa` or `hc18_train261` reads the whole 2.8 or 6.5 GB archive (about
3-10 min). CRCNS terms: cite the data set and its paper; some labs ask to be
consulted about authorship (see each loader's `DATASET["license"]`).

## Studies

`studies/` holds analyses that use the loaders but are not dataset ×
operation cells; each writes to `results/studies/`:

- `warning_calibration.py`: the numbers behind proposed warnings (samples
  exactly at the origin, fewest bins per dimension, head-direction coverage)
  on every clean recording.
- `information_bias_null.py`: `spatial_information_bias` against simulated
  untuned Poisson and bursty cells on each recording's own occupancy, and
  real units against 100 circular shifts.
- `hc11_vs_000044.py`: the same session from CRCNS hc-11 and DANDI 000044,
  unit by unit, sample by sample.
- `egocentric_boundary_survey.py`: the `rates.egocentric_boundary` operation
  on every recording with head direction, without rewriting the driver's
  results.
- `environment_fidelity.py`: whether the environment the driver builds for
  each recording has the apparatus's shape and connectivity (components,
  holes, links no sample comes near, path lengths against the distance along
  the track or the lattice's bars, the outline against a stated arena);
  figures go to `$NEUROSPATIAL_DATA_DIR/figures/environment_fidelity/`, not
  tracked. Summary in [ENVIRONMENT_FIDELITY.md](ENVIRONMENT_FIDELITY.md).

## Read the results

- `results/matrix.md`: one row per dataset, one column per operation. `ok`
  means neurospatial did it through the public API with no adaptation;
  `wk G-###` worked only with a harness workaround; `miss G-###` cannot be
  expressed with the public API; `err G-###` raised (the message is in the
  JSON, the gap id comes from `verdicts.json`);
  `–` the dataset lacks the input. `⚠N` counts distinct warning messages.
  A second table gives runtime and peak traced memory.
- `results/warnings.md`: every warning on correctly loaded data, and every
  hazard probe (the recording re-run with a known defect put back, such as
  headings in degrees or spike times in ms), each with a verdict from
  `verdicts.json`.
- `results/reference_checks.md`: neurospatial against pynapple and opexebo on
  the same arrays, with the conventions that differ.
- `results/<dataset>.json`: everything above for one dataset, plus the
  conversions the loader applied, the expectation checks and package versions.

## Add a dataset

1. Write `load_<key>.py` with `DATASET` and `EXPECTATIONS` only, register it
   in `run.py:LOADERS`, and commit before loading any data, so the
   expectations are not tuned to the result.
2. Implement `load(cache=None) -> Recording` (see `_record.py`): fetch with
   `_fetch.fetch` and a pinned SHA-256, try neurospatial's readers first, walk
   the whole NWB file before deciding something is missing, check every stored
   unit against the value ranges, and list every conversion.
3. Add `HAZARDS` for any quirk the loader corrects, so the driver can check
   whether neurospatial would have caught it.
   Where a source states the apparatus, declare it in
   `geometry["environment"]`, a zero-argument callable returning the
   environment, placed on the clean samples inside `load()`: arenas with
   `_geometry` (a stated square or circle), tracks and mazes with `_tracks`
   (a graph from the authors' file or a stated layout fitted to the samples,
   the stated width or one measured, the fit residual in `conversions`). The
   driver analyses it (`env_kind` "declared" in the matrix);
   `run.py --inferred-env` reruns on samples-built environments into
   `results/inferred_env/` for comparison
   (`studies/declared_vs_inferred.py`).
4. Run the driver, give every non-`ok` cell and every new warning or hazard row
   a gap entry or verdict, and commit the results.
