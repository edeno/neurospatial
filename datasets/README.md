# Real-data survey

Scripts that load public spatial-neuroscience recordings into plain arrays, run
a fixed set of neurospatial operations on each, and record what worked, what
needed a workaround, what neurospatial's warnings said, and where neurospatial
disagrees with another tool. The findings are in [GAPS.md](GAPS.md); the
tables are in [results/](results/). This folder is not part of the installed
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
rebuilds the Markdown tables from the JSON results. One loader alone:
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
The labyrinth's trajectories are Python pickles; its loader reads them with a
restricted unpickler that resolves only the trajectory record and int64 or
float64 NumPy arrays, and refuses every other global or dtype.

| Dataset | What it is | Size | License |
| --- | --- | --- | --- |
| `sargolini_000582` | Rat MEC grid/conjunctive cells, 1 m box (DANDI 000582) | 15.9 MB | CC-BY-4.0 |
| `peyrache_mouse32` | Mouse ADn head-direction cells, wake and sleep (OSF jb2gd) | 36.6 MB | unstated on OSF; source DANDI 000056 is CC-BY-4.0 |
| `figure8_000552` | Mouse CA1 on a figure-8 maze, ripples and sleep states (DANDI 000552) | 96.0 MB | CC-BY-4.0 |
| `rosenberg_labyrinth` | Mouse exploring a binary-tree labyrinth, behavior only (Rosenberg et al. 2021 repository) | 6.3 MB | MIT |
| `yartsev_bat_demo` | Bat CA1 during 3-D flight (Zenodo 15738988; members of a 3.3 GB zip read by byte range) | 260 MB | CC-BY-4.0 |

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
4. Run the driver, give every non-`ok` cell and every new warning or hazard row
   a gap entry or verdict, and commit the results.
