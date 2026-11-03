# Designs

[← back to PLAN.md](PLAN.md)

- [Streamed NWB](#streamed-nwb)
- [Safe pickles](#safe-pickles)
- [MATLAB v7 files](#matlab-v7-files)
- [Information-bias null study](#information-bias-null-study)
- [Gated evidence](#gated-evidence)
- [New operation columns](#new-operation-columns)

## Streamed NWB

DANDI 000044 and 000447 files are 5–9 GB, mostly raw LFP. The survey needs
position, units, epochs and intervals: a few MB. Reading them by HTTP byte
range was checked at planning time (2026-11-02): opening each of the eight
000044 files with `fsspec` + `h5py` and listing `processing/behavior` and the
epochs took about 10 s per file.

Integrity: a whole-file SHA-256 cannot be computed without downloading the
file, but DANDI publishes one per asset. `GET
https://api.dandiarchive.org/api/assets/<asset_id>/` returns
`digest["dandi:sha2-256"]` (checked for asset
`185b8a36-d671-4688-ba05-9e89a902c486`). The loader pins that digest, as it
pins a download's SHA-256 today, and refuses the stream if the API reports a
different one.

Add to `datasets/_fetch.py`:

```python
def dandi_stream_url(asset_id: str, *, sha256: str) -> str:
    """Return the asset's download URL after checking DANDI's published SHA-256.

    For files read by byte range, where hashing the whole file would mean
    downloading it.
    """
    with urllib.request.urlopen(f"https://api.dandiarchive.org/api/assets/{asset_id}/") as r:
        published = json.load(r)["digest"]["dandi:sha2-256"]
    if published != sha256:
        msg = (
            f"DANDI asset {asset_id} now has SHA-256 {published}, not the pinned"
            f" {sha256}; the file changed, so recheck the loader before updating the pin"
        )
        raise ValueError(msg)
    return dandi_asset_url(asset_id)


@contextlib.contextmanager
def open_nwb(source: str) -> Iterator[Any]:
    """Yield the NWBFile at a local path, or at an https URL read by byte range."""
    import pynwb

    if not source.startswith(("https://", "http://")):
        with pynwb.NWBHDF5IO(source, "r", load_namespaces=True) as io:
            yield io.read()
        return
    import fsspec
    import h5py

    with (
        fsspec.open(source, "rb", block_size=4 << 20, cache_type="blockcache") as remote,
        h5py.File(remote, "r") as h5,
        pynwb.NWBHDF5IO(file=h5, mode="r", load_namespaces=True) as io,
    ):
        yield io.read()
```

Rename `params["nwb_path"]` to `params["nwb_source"]` everywhere it appears
(`_record.py:52`, the six NWB loaders, `operations.py:503`,
`reference_checks.py:298`, `load_miniscope_002030.py:110`) and route every
open through `open_nwb`. Downloaded loaders store `str(path)`; streamed
loaders store the URL from `dandi_stream_url`.

Streamed loaders read the few datasets they need with `h5py` paths inside
`open_nwb`'s file (or `pynwb` objects, whose `.data[:]` reads only that
dataset), never `nwbfile.acquisition[...]` on raw `ElectricalSeries` data.
The smoke test records bytes read (wrap `remote` and count `read` sizes, or
time it) and wall time per loader; a loader that reads more than about
500 MB is reworked before the full run.

## Safe pickles

Gardner 2022's `.npz` members need `allow_pickle=True`, which runs arbitrary
code from the file. Read them without that:

1. Open the `.npz` as a zip (`zipfile.ZipFile`); for each `<name>.npy`
   member, read the header with `numpy.lib.format.read_magic` and
   `read_array_header_1_0` / `read_array_header_2_0`.
2. Non-object dtype: `numpy.lib.format.read_array(member, allow_pickle=False)`.
3. Object dtype (NumPy writes these as a pickle after the header): first list
   the globals the stream would import, without executing it:

   ```python
   def pickle_imports(blob: bytes) -> set[str]:
       """Every GLOBAL/STACK_GLOBAL a pickle would resolve, read without executing it."""
       found, recent = set(), []
       for op, arg, _ in pickletools.genops(blob):
           if op.name == "GLOBAL":
               found.add(arg.replace(" ", "."))
           elif op.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
               recent = [*recent[-1:], arg]
           elif op.name == "STACK_GLOBAL":
               found.add(".".join(recent) if len(recent) == 2 else "<memoized>")
       return found
   ```

   Write the observed set into the loader's docstring, then unpickle with an
   allow-list unpickler modelled on `_TrajUnpickler`
   (`datasets/load_rosenberg_labyrinth.py:117-133`). Use the same
   restricted `numpy.dtype` factory (`:103-114`), extended only to the dtypes
   the inspection found. A `<memoized>` entry means the inspection could not
   name a global: stop and inspect by hand before allowing anything.
4. Any global outside the allow-list raises `pickle.UnpicklingError`.

The labyrinth's unpickler stays as it is. Do not factor the two into a shared
helper unless a third pickled dataset appears.

## MATLAB v7 files

Vollan 2024/25 files are MATLAB v7 `.mat` (not HDF5), and their documentation
(`sharing_v4/documentation/mat_files.md`) shows MATLAB `string` arrays and a
`Session` class object. `scipy.io.loadmat` reads numeric fields but returns
opaque `MatlabOpaque` objects for `string` and class instances. Read with
`mat-io` (`matio.load_from_mat`, already in the `datasets` group), which reads MATLAB
objects such as `table` (used for the bat demo); whether it reads `string`
and `Session` objects is unverified. Check that first on
the 110 MB linear-track file. If it fails on those objects, read the numeric
fields with `scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)`,
recover unit ids from whatever form the `string` array takes (or number units
by position and say so in `conversions`), and log the reader gap as a data
quirk.

## Information-bias null study

Question (FINDINGS decision 3): on sparse occupancy, does
`spatial_information_bias()` (`src/neurospatial/encoding/spatial.py:1555`;
derivation in its Notes; batch form `_information_bias_terms`,
`src/neurospatial/encoding/_metrics.py:578`) predict the information an untuned
cell shows? The survey's bat result (median 3.38 bits/spike, shifted 3.55,
bias 0.37) says the shift null is far above the bias estimate. Two
explanations must be separated:

- **Sparse occupancy breaks the second-order expansion** (an estimator
  problem that a correction or a sparse-occupancy warning addresses).
- **Non-Poisson temporal structure** (bursts, flight-locked firing): the
  bias is the expected information of an untuned *Poisson* cell, and circular
  shifts keep the spike train's temporal structure. Then the estimate is right
  for what it states, and the gap is the null model, not sparseness.

Script `datasets/studies/information_bias_null.py`, writing
`datasets/results/information_bias_null.md` and `.json`. All seeds fixed and
recorded.

1. **Simulated, untuned Poisson.** Build occupancy from the bat's own
   trajectory (`load_yartsev_bat_demo.load()`), so sparseness is real, and
   from a dense 2-D control (000582's trajectory). For mean rates
   {0.5, 2, 8} Hz, draw 200 untuned Poisson trains per rate (spike counts
   per sample `rng.poisson(rate * dt)`, times at the sample), compute
   `compute_spatial_rates(...)` with the operation's settings
   (`datasets/operations.py:64` `_rates`), and record, per rate, the mean
   `spatial_information()` and the mean `spatial_information_bias()`.
   Ratio mean SI / mean bias near 1 means the estimate holds on sparse
   occupancy; far from 1 means it breaks there.
2. **Simulated, untuned but bursty.** Same, with spikes in bursts (a
   Poisson process of burst onsets, each with a geometric number of spikes,
   mean 3, 5 ms apart) at the same mean rates. Measures how much temporal
   structure alone raises information above the Poisson bias.
3. **Real units.** For the bat, 000582, Gardner and the 000044 circular track:
   100 circular shifts per unit (the rule in `information_vs_shifts`,
   `datasets/operations.py:153-170`, with seeds 0–99). Per unit: real SI,
   shift-null mean, shift-null 95th percentile, and the bias estimate. Report
   per dataset the median of null mean / bias, and the share of units with
   `SI - bias` above the classify cutoff but SI below the shift 95th
   percentile (units `classify()` would pass that the shuffle rejects).
4. **Occupancy summary.** Per dataset: share of visited bins under 0.1 s, the
   median seconds per visited bin, and the median spikes per visited bin.

Reading the result (write it in the report, do not change neurospatial):

| Step 1 ratio (bat) | Step 2 / step 3 | What it means |
| --- | --- | --- |
| ≈ 1 | bursty ≫ Poisson; null mean ≫ bias | The estimate is right; the gap is temporal structure. A sparse-occupancy warning would be the wrong fix; the docs should say the bias is a Poisson null |
| ≫ 1 | — | The estimate breaks on sparse occupancy: the agreed warning (or a correction) applies |

Runtime: 100 shifts × units × `compute_spatial_rates`. The 4-shift operation
takes seconds per dataset (`results/matrix.md` runtime table), so 100 shifts
are minutes; smoke-test one dataset first and record the time.

## Gated evidence

Phase 1 ends by adding one section per gated decision to
`datasets/FINDINGS.md`, each a table plus a recommendation. The maintainer
decides; nothing is implemented.

**Grid score (decision 2).** Run `grid_score_check`
(`datasets/reference_checks.py:220`) on every grid-module recording (Gardner
modules; Vollan open field with the authors' module labels), plus option (a)
computed in the harness: a NaN-aware Pearson correlation at every lag with at
least `k` overlapping visited bins, `k` in {20, 1/4 of visited bins}, and score
it with `neurospatial.encoding.grid_score`. Table per dataset:
n units, r and max |difference| of neurospatial vs opexebo, of (a) vs
opexebo, of zero-filled (option c) vs opexebo, and for Vollan the correlation
of each with the authors' published per-unit or per-module gridness
(`unitAcorrClus` / `medianGridScore`). Add the number of units whose verdict
at the paper's own cutoff changes between conventions. Option (b) (relax the
quarter-overlap rule) is option (a) with `k` = the relaxed threshold, so one
code path covers both.

**Origin share (decision 4, check 1).** For every clean recording with
positions (old and new), the share of samples exactly at the origin, both for
the positions the loader returns and, where the file has a linearized
position, for the linearized coordinate. The check's threshold is 0.1 %; the
table shows which clean recordings would trip it. The start-of-track case is
000044's and 000447's linearized positions (0 is the start of the track); if
those trip it, say so: the check must then exclude 1-D linearized input or use
a different rule.

**Heading coverage (decision 1, check 3).** On Vollan's linear track: 36
direction bins, count bins holding ≥ 0.5 % of samples, for all samples and
for running samples only. The proposed threshold is fewer than half the bins
(18). Also report the same count on every new recording with head direction.

## New operation columns

Added only when a batch brings the input; each is one function in
`datasets/operations.py` and one `Operation(...)` row in `OPERATIONS`
(`:682`), following the existing pattern (return `Outcome`; `missing` with a
gap id when the public API cannot express it).

- `io.read_pose` (phase 2, DANDI 001845): `neurospatial.io.nwb.read_pose`
  (`src/neurospatial/io/nwb/_pose.py:42`) on the file; record keypoint
  names, NaN share per keypoint, and whether the result feeds
  `compute_spatial_rates` (via a chosen keypoint) without a workaround.
- `rates.object_vector` (phase 2, Kanter 2025 if object positions are in the
  file): `neurospatial.encoding.compute_object_vector_rates(env, spike_times,
  times, positions, object_positions, ...)` (`tests/data/public_api.txt:234`).
- `rates.egocentric_boundary` (phase 3, Zhang 2022):
  `neurospatial.encoding.compute_boundary_vector_rates(env, spike_times, times,
  positions, headings, *, direction_frame="egocentric", ...)`
  (`tests/data/public_api.txt:226`; check the accepted `direction_frame`
  values in its docstring). Compare its tuning peaks with the dataset's own
  classification if the files carry one.
