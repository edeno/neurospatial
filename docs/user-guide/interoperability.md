# Interoperability

neurospatial is **array-first**. Every analysis in the library runs on plain
NumPy arrays and needs **no optional dependencies** — you build an
[`Environment`](environments.md), pass spike times / timestamps / positions as
arrays, and get results back as arrays (or result objects that wrap them).

Loaders and simulators return frozen holders with named attributes. Pass
those attributes explicitly to analysis functions; timestamps always precede
positions. The pynapple adapters, [`SpikeTrains`](#spiketrains-and-epoch-selection)
and [`BayesianDecoder`](#bayesiandecoder) preserve unit identity at array handoffs.

!!! info "The array path is always available"
    `import neurospatial` never imports `pynapple` or `pynwb`. The array path is
    byte-for-byte identical whether or not the optional extras are installed —
    the extras only add adapters at the boundary.

Install the optional extras only when you need the corresponding adapters:

```bash
pip install neurospatial[pynapple]   # pynapple TsGroup / Tsd / IntervalSet adapters
pip install neurospatial[nwb]        # NWB read/write (pynwb)
```

## Array-first stays primary

Everything below runs on plain arrays with a bare `pip install neurospatial`.
The example computes rate maps and decodes position from arrays, then
uses the same arrays with a fitted decoder.

<!-- docs-test: run -->
```python
import numpy as np

from neurospatial import Environment
from neurospatial.encoding import compute_spatial_rates
from neurospatial.decoding import BayesianDecoder, decode_session, decoding_error
from neurospatial.simulation import (
    PlaceCellModel,
    generate_population_spikes,
    simulate_trajectory_ou,
)

# --- Array-first: plain NumPy arrays, no optional dependencies -------------
# (In a real analysis, load your own env, spike_times, times, and positions.)
env = Environment.from_samples(
    np.linspace(0.0, 100.0, 51).reshape(-1, 1), bin_size=2.0
)
env = env.with_metadata(units="cm")
positions, times = simulate_trajectory_ou(
    env, duration=120.0, dt=0.02, speed_mean=15.0, seed=0, speed_units="cm"
)
cells = [
    PlaceCellModel(env, center=np.array([c]), width=10.0, max_rate=20.0, seed=i)
    for i, c in enumerate(np.linspace(5.0, 95.0, 15))
]
spike_times = generate_population_spikes(
    cells, times, positions, seed=0, show_progress=False
)

# Rate maps and a one-call decode, straight from arrays.
rates = compute_spatial_rates(env, spike_times, times, positions)
result = decode_session(env, spike_times, times, positions, dt=0.1)
actual = np.interp(result.times, times, positions[:, 0]).reshape(-1, 1)
print(f"array-first median error: "
      f"{np.nanmedian(decoding_error(env, result.map_position, actual)):.1f} cm")

# --- BayesianDecoder: the same decode as an object ------------------------
decoder = BayesianDecoder(env, dt=0.1).fit(spike_times, times, positions)
prediction = decoder.predict(spike_times, times)
# Equal here because no position is missing (see the BayesianDecoder section).
assert np.array_equal(prediction.posterior, result.posterior)
error = decoder.score(spike_times, times, positions, metric="median_error")
print(f"BayesianDecoder.score median error: {error:.1f} cm")
```

## Installed snapshot adapters

An external package can supply an installed `SnapshotAdapter` object that
converts its data into the [explicit binning records](environments.md#bring-explicit-bins).
Publication validates and owns the resulting built-in data. It retains no
provider callbacks and does not register custom query algorithms or builders.

| Role | Required result |
| --- | --- |
| `"locator"` | A built-in locator, such as `NearestBinLocator` |
| `"track"` | `TrackLinearization` |
| `"arena"` | Built-in Arena data |
| `"binning"` | `Binning` with supported built-in components |

For example, a package module `my_lab.adapters` could expose this object.
`snapshot` and `decode` must be pure: they return data without changing their
inputs. `provider_type` matches the exact external type; subclasses need their
own declaration.

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial.binning import NearestBinLocator
from neurospatial.extensions import SnapshotAdapter

class RecordedSites:
    def __init__(self, centers):
        self.centers = np.asarray(centers, dtype=float)

def snapshot_sites(provider):
    return NearestBinLocator(provider.centers, max_distance=0.25)

def decode_sites(metadata, arrays, prefix):
    return NearestBinLocator(
        arrays[f"{prefix}/centers"], max_distance=metadata["max_distance"]
    )

adapter = SnapshotAdapter(
    name="my_lab.recorded_sites",
    version=1,
    role="locator",
    provider_type=RecordedSites,
    snapshot=snapshot_sites,
    decode=decode_sites,
)
assert adapter.role == "locator"
provider = RecordedSites([[0.0, 0.0], [2.0, 0.0]])
snapshot = adapter.snapshot(provider)
np.testing.assert_array_equal(snapshot.locate(np.array([[0.0, 0.0]])), [0])
restored = adapter.decode(
    {"max_distance": 0.25}, {"sites/centers": provider.centers}, "sites"
)
np.testing.assert_array_equal(restored.locate(np.array([[2.0, 0.0]])), [1])
```

Declare the object in that package's `pyproject.toml`, then install the package:

```toml
[project.entry-points."neurospatial.snapshot_adapters.v1"]
recorded_sites = "my_lab.adapters:adapter"
```

The entry point exposes an object, rather than a factory. Its group versions
the adapter API; `version` identifies the custom stored payload that `decode`
accepts. Custom records use the adapter's `name` as `codec` and its version as
`codec_version`. The decoder receives the supplied array namespace and prefix.
Names must not collide with built-in codecs or other installed adapters; only
one adapter may handle a given provider type and role.

The package is required to publish its external providers or load its original
custom records. After loading and saving an environment, the saved data use
canonical built-in codecs and can be loaded without that package. Missing
adapters and unsupported versions raise with installation guidance.

Use the public contract helper with boundary answers derived independently
from the intended geometry. It checks lookup, outside sentinels, ownership,
missing-face refusal and copy/storage round trips. This native grid example
specifies the internal boundary owner (the bin above a shared face), the outer
edge (which belongs to the last bin) and a point outside:

<!-- docs-test: run -->
```python
import numpy as np
from neurospatial import Environment
from neurospatial.testing import assert_environment_contract

env = Environment.from_grid_mask(
    np.ones((2, 1), dtype=bool),
    grid_edges=(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])),
)
assert_environment_contract(
    env,
    points=[[0.0, 0.5], [1.0, 0.5], [2.0, 0.5], [-0.1, 0.5]],
    expected_bins=[0, 1, 1, -1],
)
```

Run the same helper on environments published by an installed adapter, using
that geometry's own hand-checked boundary and outside points.

## Simulator attributes feed analyses

A `SimulationSession` carries `env`, `spike_times`, `unit_ids`, `times`,
`positions`, `models`, integer-label `ground_truth` and `metadata`. It is
immutable: its arrays are read-only, `spike_times` and `models` are tuples, and
`ground_truth` and `metadata` are read-only mappings; build a changed session
with `dataclasses.replace`. Spike arrays, labels and models must align
one-to-one. Unit labels also select the ground truth for validation and the
rate panels for plotting.

<!-- docs-test: run -->
```python
from neurospatial.encoding import compute_spatial_rates
from neurospatial.simulation import open_field_session, validate_simulation

sim = open_field_session(duration=60, n_place_cells=5, seed=0)
rates = compute_spatial_rates(
    sim.env, sim.spike_times, sim.times, sim.positions, unit_ids=sim.unit_ids,
)
assert set(sim.ground_truth) == set(sim.unit_ids)
print(rates.summary_table().index.tolist())
validation = validate_simulation(sim, unit_ids=[0, 3, 4])
print(validation.summary_table()[["center_error", "correlation", "status"]])
assert validation.unit_status == ("failed", "passed", "failed")
```

`validate_simulation` returns one row per selected unit; one minute is too
short to recover every field, so units 0 and 4 fail the default center-error
limit here. `plot_session_summary(sim, unit_ids=[1, 3])` uses the same label
selection.
Analysis functions take arrays; the validation and summary plot functions
need the simulator holder because it carries simulation-specific data.

## `SpikeTrains` and epoch selection

`SpikeTrains` is a frozen bundle of ragged per-unit spike trains plus their
identity labels (`unit_ids`) and an optional per-unit metadata table
(`unit_table`). It gives you positional access, label selection, iteration and
a metadata-driven `filter`, and it flows **directly** into the batch encoding / decoding functions
(it duck-types as a spike-input group, so its `unit_ids` are carried into the
result).

```python
import numpy as np
import pandas as pd
from neurospatial.encoding import SpikeTrains

st = SpikeTrains(
    [np.array([0.1, 1.5, 2.9]), np.array([0.5, 3.0, 6.0])],
    unit_ids=np.array([7, 9]),
    unit_table=pd.DataFrame({"region": ["CA1", "CA3"], "quality": [0.9, 0.4]}),
)

st.index          # unit_ids (the group-key surface): array([7, 9])
list(st)          # iterate -> the per-unit train arrays
st[1]             # by position -> array([0.5, 3. , 6. ])
st.sel(unit_id=9) # by label: the same unit

# Metadata-driven selection (returns a new SpikeTrains).
ca1 = st.filter("region == 'CA1' and quality > 0.5")

# Flows straight into batch compute — unit_ids are carried into the result.
rates = compute_spatial_rates(env, st, times, positions)
rates.sel(unit_id=9)  # that unit's rate map; rates[1] is the same unit
```

### Restricting to epochs

For occupancy, firing rates, movement metrics and decoding, keep the original
arrays and pass `epochs=` directly to each analysis. This preserves excluded
intervals even when they are shorter than `max_gap`. For example:

<!-- docs-test: run -->
```python
from neurospatial import compute_spatial_rates

run_epochs = np.array([[1.0, 4.0], [7.0, 9.0]])
rates = compute_spatial_rates(
    env, spike_times_list, times, positions, epochs=run_epochs,
)
print("Analyzed windows:", rates.analysis_settings["epochs"])
print("Observed time (s):", rates.occupancy.sum())
```

Spatial, directional, view and object-vector rate results all carry
`analysis_settings` and `spatial_support`. The settings record the resolved
windows, the intervals actually retained after filtering, and the estimator's
choices. The support mask identifies bins with positive, sufficient occupancy
and finite rates; its coordinates follow the result (locations, angles or
polar object vectors). Smoothing can predict a finite rate outside observed
coverage, so use the support mask when an analysis requires observed bins.
These fields survive population indexing, DataFrame and xarray export; xarray
stores the settings as JSON for NetCDF compatibility. Manually constructed
results default to `analysis_settings=None`.

Spatial and view smoothing, and smoothed object-vector maps, share a physical
bandwidth default: 5 cm converted to declared metric units, or 2.5 median
graph-edge lengths for unspecified/custom units. Object-vector results inherit
the supplied environment's distance units; their angles remain radians.
Directional smoothing keeps its explicit angular bandwidth, and object-vector
`method="binned"` applies no smoothing. Supply explicit distance ranges and gaze
distances in your coordinate units.

`restrict`, `in_epochs`, and `restrict_spike_trains` select time windows.
`restrict(times, *arrays, epochs=...)` slices `times` and any number of arrays
**aligned to it** by the same in-epoch mask; `in_epochs(t, epochs)` returns the
boolean mask; and `restrict_spike_trains(spike_times, epochs)` masks *ragged* trains
(each unit by its own timestamps).

```python
from neurospatial.behavior import restrict, in_epochs, restrict_spike_trains

run_epochs = np.array([[1.0, 4.0], [7.0, 9.0]])   # (n_intervals, 2)

# Slice aligned arrays (position samples share one time axis).
t_kept, pos_kept = restrict(times, positions, epochs=run_epochs)

# Boolean mask over timestamps.
mask = in_epochs(times, run_epochs)

# Ragged per-unit spikes: each train masked by its own timestamps.
kept = restrict_spike_trains(st, run_epochs)
```

`epochs` accepts a `(start, stop)` pair, an `(n, 2)` array or list of interval
rows (including `[[0, 5], [10, 15]]`), or a pynapple `IntervalSet`. Parallel
start/stop arrays are rejected: convert them with `np.column_stack([starts, stops])`.

!!! warning "Sliced samples do not retain observation windows"
    `restrict()` warns when aligned arrays would join separate windows. Passing
    those arrays to an analysis without `epochs=` can reintroduce excluded time.
    Prefer the original arrays with `epochs=`; if using sliced arrays, pass the
    windows again. Point selection uses the same `[start, stop)` windows as
    the analyses, so adjacent train/test windows never share a spike or
    event. A pynapple `IntervalSet` includes its end, and its rows are
    converted so a point at `end` stays inside; for plain arrays of
    pynapple bounds, pass `closed="both"` to match pynapple's `restrict`.

## Analysis settings and exports

Spatial rate results retain `analysis_settings`: selected epochs, the intervals
actually observed after gating, spike-recording windows, speed source and
threshold, gap threshold, occupancy threshold, smoothing and estimator choices.
`spatial_support` marks observed bins meeting the occupancy threshold with finite
rates. Indexing a population result preserves these settings.

<!-- docs-test: run -->
```python
from neurospatial import compute_spatial_rates

rates = compute_spatial_rates(
    env, spike_times_list, times, positions, epochs=(1.0, 4.0),
    bandwidth=5.0, min_occupancy=0.05,
)
table = rates.to_dataframe()
assert table.attrs["analysis_settings"]["epochs"] == [[1.0, 4.0]]
print(table.attrs["units"])
print("Supported bins:", int(rates.spatial_support.sum()))
```

`to_xarray()` stores these settings as JSON in `attrs["analysis_settings"]`,
alongside a `spatial_support` variable; they survive NetCDF export.
`write_spatial_rates` stores them in NWB too (spatial-rates schema 2.5);
`read_spatial_rates` refuses records written with an earlier schema, so
recompute and rewrite those. Results you construct by hand without these choices
have `analysis_settings=None`. CSV does not preserve
DataFrame attributes: save the settings as a JSON sidecar, or use NetCDF/NWB.

## `BayesianDecoder`

`BayesianDecoder` is an optional **object wrapper** over the functional
[`decode_session`](../api/index.md) path. It is frozen: `fit(...)` builds the
encoding models and returns a **new** fitted decoder, `predict(...)` returns a
`DecodingResult`, `predict_summary(...)` returns a memory-safe `DecodingSummary`,
and `score(...)` returns a scalar decode error.

```python
from neurospatial.decoding import BayesianDecoder

decoder = BayesianDecoder(env, dt=0.1)
decoder.is_fitted                       # False

fitted = decoder.fit(spike_times, times, positions)   # returns a NEW decoder
fitted.is_fitted                        # True

result = fitted.predict(spike_times, times)                  # DecodingResult
summary = fitted.predict_summary(spike_times, times, time_chunk=1024)
error = fitted.score(spike_times, times, positions,
                     metric="median_error", distance="euclidean")

# Train/test split: fit on one epoch, evaluate on another.
fitted = decoder.fit(spike_times, times, positions, epochs=(0.0, 60.0))
```

On a recording without missing tracking, `fit(...).predict(...)` on the same
inputs and parameters equals `decode_session` exactly. With NaN positions they
differ: `decode_session` does not decode tracking losses longer than `max_gap`,
while `predict`, which takes no positions, decodes them. `decode_session` fits
and decodes the same data, an in-sample decode; fit and predict on separate
epochs to measure accuracy on held-out data.

Because decoding runs through the `Environment`, `BayesianDecoder` decodes
**linearized tracks**, masked open fields, and graph-based layouts — not just a
rectangular grid — and `score(..., distance="geodesic")` measures error along the
environment's connectivity graph.

## pynapple interop

pynapple objects convert to and from plain arrays **at the boundary** with
`from_pynapple` / `to_pynapple`.

!!! note "Requires the `pynapple` extra"
    `pip install neurospatial[pynapple]`. Only these two adapter functions
    import pynapple, and they import it lazily.

A pynapple object is defined only on its `time_support`: the recording epochs,
with gaps between them for artifacts, sleep boxes or lost tracking.
`from_pynapple` returns that support as `epochs`, a `(n_epochs, 2)` array of
half-open `[start, stop)` rows in seconds, so you can pass it to the analysis and keep
the gaps out of occupancy and rates:

```python
from neurospatial.encoding import compute_spatial_rates
from neurospatial.io import from_pynapple, to_pynapple

# Ingress: pynapple -> plain arrays plus their time support.
times, positions, epochs = from_pynapple(tsdframe)   # TsdFrame / Tsd
# A TsGroup's support is the recording coverage only if it was built with the
# acquisition intervals; by default it runs from the first to the last spike.
trains, unit_ids, recorded = from_pynapple(tsgroup)  # TsGroup
spike_times, unit_epochs = from_pynapple(ts)         # one unit's Ts
run_epochs = from_pynapple(intervalset)              # IntervalSet -> (n, 2)

rates = compute_spatial_rates(
    env, trains, times, positions,
    epochs=epochs, spike_window=recorded, unit_ids=unit_ids,
)
```

Without `epochs=`, a gap shorter than `max_gap` (1 s by default) is counted as
time spent at the last position before it, which inflates occupancy in that bin.

Positions come back in the frame's column order. A `TsdFrame` whose columns are
the axis names in another order, such as `["y", "x"]`, is refused rather than
silently swapping the axes; reorder it with `tsdframe.loc[["x", "y"]]` first.

pynapple intervals include their end, while neurospatial windows are half-open
`[start, stop)`. Each pynapple `end` becomes the next float above it
(`np.nextafter(end, np.inf)`), so the epochs hold exactly the timestamps pynapple
holds: a sample or spike at an epoch end, such as the last spike of a `TsGroup`'s
default support, stays inside. An `IntervalSet` passed directly as `epochs=` or
`spike_window=` is converted the same way, and `to_pynapple` converts back.

A raw `TsGroup` can also supply spikes and unit labels directly, and an
`IntervalSet` is accepted directly as `epochs=` or `spike_window=`:

```python
from neurospatial.decoding import decode_session

result = decode_session(
    env, tsgroup, times, positions, dt=0.1, spike_window=recording_intervals
)
```

Passed directly, a `TsGroup` contributes only its spike times and labels; its
`time_support` is ignored. Without `spike_window=`, spikes are assumed to have
been recorded wherever position was tracked. Use the group's time support as
`spike_window=` only if you set it to the true acquisition intervals: a
`TsGroup` built from spike times alone gets a default support that runs from
its first spike to its last, which is not recording coverage.

Egress turns a decoded MAP track back into a pynapple `Tsd` / `TsdFrame`. Its
time support is the decode's `spike_window`, so decoding over several windows
stays several intervals instead of one interval spanning the gaps:

```python
tsdframe = to_pynapple(result)                              # DecodingResult
tsd = to_pynapple(times, linear_position, epochs=epochs)    # plain arrays
```

A decode result does not record the `epochs=` it was decoded over; if you
restricted decoding that way, pass the same windows as `to_pynapple(result,
epochs=...)`.

## NWB interop

The NWB adapters read population spikes, position, and pose out of an NWB file,
and round-trip a `SpatialRatesResult` back into one.

!!! note "Requires the `nwb` extra"
    `pip install neurospatial[nwb]`. As with pynapple, `import neurospatial`
    never imports `pynwb`; the readers import it only when called.

Read components explicitly. These readers return frozen, non-iterable holders:

| Reader | Holder attributes |
| --- | --- |
| `read_position` | `NWBPosition.times`, `.positions`, `.units`, `.frame` |
| `read_head_direction` | `NWBHeadDirection.times`, `.headings` (radians), `.frame` |
| `read_pose` | `NWBPose.bodyparts`, `.times`, `.skeleton`, `.units`, `.frame`, `.confidence` |
| `read_units` | `NWBUnits.spike_times`, `.unit_ids`, `.obs_intervals`, `.spike_window` |

When a file holds more than one matching series or container, for example raw
pixel and calibrated position, the reader raises and lists their paths instead
of picking one. Pass the name or full path to `position_name=`,
`compass_name=` or `pose_estimation_name=`.

Position values already include the stored conversion and offset. Recognized
unit aliases become `m`, `cm`, `mm`, `um` or `px` without scaling those values again.
Other nonempty declarations remain visible; `None` means no unit was declared.
Choose the physical unit explicitly in that case before creating an environment:
`environment_from_position` refuses an undeclared unit unless you pass `units=`,
and refuses `units=` or `frame=` values that conflict with the series
declarations.
Position reads also preserve nonempty `SpatialSeries.reference_frame`
descriptions verbatim in `.frame`; `None` means no frame was declared. The
`position_overlay_from_nwb` adapter forwards both units and frame and rejects
keyword overrides that would erase or change a known declaration. Supply missing
metadata explicitly, or transform the coordinates before using a different
frame or physical unit. Equivalent unit spellings do not require conversion.

Head direction angles stored with unit `radians` or `degrees` are returned in
radians; any other unit, including pynwb's default `meters`, raises because
the angles could be either. `.frame` carries the series reference frame
verbatim. The headings are assumed to be zero East and counterclockwise; if
`.frame` says otherwise (for example "0 = North, clockwise"), convert with
`np.pi / 2 - hd.headings`.

`read_pose` returns an `NWBPose` and checks that every bodypart
uses the same sample clock, units and reference frame. Untracked keypoints stay
NaN, and low-confidence keypoints are returned unchanged with their scores in
`.confidence`; choose a threshold for your tracker and mask them yourself. Different clocks must be
aligned explicitly before combining them into a pose. The
`bodypart_overlay_from_nwb` adapter carries the declared units and reference
frame into `BodypartOverlay`, which checks them against the environment. Convert
coordinates explicitly when these declarations differ; supplying a different
metadata label does not perform that conversion. Lazy pose reads validate the
clocks without materializing the coordinate arrays.

### Population fields, decoding and a truthful overlay

This recipe reads eager arrays from `session.nwb`, selects the named `epochs`
table's rows tagged `run`, and uses those windows in both encoding and decoding.
Adapt the table name, tags and 5-unit bin size to your experiment. Epochs are
an explicit analysis choice; readers do not infer them from spike times.

`units.spike_window` is acquisition coverage: the intersection of observation
intervals for the units selected by `read_units(..., unit_ids=...)`. It can differ
from the chosen analysis epochs. If the file has no `obs_intervals` column, it
is `None` and analysis results report assumed spike coverage. A recorded empty
intersection stays empty and analyses reject it; read units with different
coverage in separate calls if their shared window is too short.

<!-- nwb-docs-test: run -->
```python
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO

from neurospatial import Environment, compute_spatial_rates
from neurospatial.decoding import BayesianDecoder
from neurospatial.encoding import SpikeTrains
from neurospatial.io.nwb import read_intervals, read_position, read_units

with NWBHDF5IO("session.nwb", "r") as io:
    file = io.read()
    units = read_units(file)  # Or unit_ids=[7, 11] to select table labels.
    pos = read_position(file)
    epoch_table = read_intervals(file, interval_name="epochs")
    chosen = epoch_table["tags"].map(lambda tags: "run" in tags)
    epochs = epoch_table.loc[chosen, ["start_time", "stop_time"]].to_numpy(dtype=float)

# Eager arrays and interval metadata remain usable after the file closes.
position_units = pos.units
if position_units is None:
    raise ValueError("Choose position_units explicitly from the experiment's physical units.")
env = Environment.from_samples(pos.positions, bin_size=5.0, units=position_units)
cells = SpikeTrains(units.spike_times, unit_ids=units.unit_ids)
rates = compute_spatial_rates(
    env, cells, pos.times, pos.positions,
    epochs=epochs, spike_window=units.spike_window,
)
rate_table = rates.summary_table()
print(rate_table)

# Reuse the maps; the labelled trains meet their own cells' maps.
decoder = BayesianDecoder.from_rates(rates, dt=0.2)
result = decoder.predict(
    cells, pos.times, epochs=epochs, spike_window=units.spike_window,
)
print(result.summary())

# Align the original tracking and preserve gaps through the plotting API.
ax = result.plot(
    show_map=True, colorbar=True,
    true_times=pos.times, true_positions=pos.positions, max_gap=decoder.max_gap,
)
ax.legend()
plt.show()
```

The posterior plot's y-axis is spatial-bin indices; physical positions belong
in separate position plots or accuracy metrics. The actual overlay follows the
same display coordinates as the posterior and MAP line, with breaks across
missing tracking and recording gaps. See the
[decoder plotting recipe](workflows.md#overlaying-actual-position-on-a-posterior)
for the continuous/gapped-clock comparison. If an environment was already
persisted in the file, use `read_environment(file)` inside the `with` block
instead of inferring bins, with its coordinates matching the position channel.

`read_units`, `read_position`, and `read_pose` accept `lazy=True` for large files.
Lazy position reads require identity conversion and offset.

!!! warning "Lazy handles are only valid while the file is open"
    Slice or materialize lazy arrays inside the `with NWBHDF5IO(...)` block.
    Eager arrays (the default), IDs and observation-window metadata remain
    valid after close.

```python
with NWBHDF5IO("session.nwb", "r") as io:
    units = read_units(io.read(), lazy=True)
    first_unit = np.asarray(units.spike_times[0])
```

### Round-tripping rate maps

`write_spatial_rates` persists a population `SpatialRatesResult` — the per-unit
firing-rate maps, shared occupancy, `unit_ids`, optional `unit_table`, and a
connected copy of the `Environment` — and `read_spatial_rates` reconstructs an
equal result. Because the environment round-trips with its connectivity intact,
graph operations work on the restored env with no `env=` argument.

```python
from pynwb import NWBHDF5IO
from neurospatial.encoding import compute_spatial_rates
from neurospatial.io.nwb import write_spatial_rates, read_spatial_rates

rates = compute_spatial_rates(env, spike_times, times, positions)

# Write.
with NWBHDF5IO("session.nwb", "r+") as io:
    nwbfile = io.read()
    write_spatial_rates(nwbfile, rates, name="ca1_place_fields")
    io.write(nwbfile)

# Read back (env restored from the file; an env= you pass must equal it).
with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()
    restored = read_spatial_rates(nwbfile, name="ca1_place_fields")
    restored.firing_rates.shape   # (n_units, n_bins)
```

The record is a table in the processing module `analysis`, plus its occupancy
(seconds) and a saved copy of the environment. To replace a record, open the
file with `"r"`, call `write_spatial_rates(..., overwrite=True)`, and save a
copy with `NWBHDF5IO(new_path, "w").export(src_io=io, nwbfile=nwbfile)`. pynwb
cannot delete objects already written to a file opened with `"r+"`, so
`overwrite=True` refuses such files.

## See Also

- **[Complete Workflows](workflows.md)**: End-to-end encode / decode examples
- **[Spatial Analysis](spatial-analysis.md)**: Occupancy, fields, and trajectory operations
- **[API Reference](../api/index.md)**: NWB holders, `SpikeTrains`, `BayesianDecoder`, and the interop adapters
- **[Loading from NWB notebook](../examples/09_nwb_io.ipynb)**: A worked NWB read example

## Next Steps

- **Read components**: select units and epochs explicitly, then pass holder
  attributes and acquisition coverage to your analysis.
- **Filter and restrict**: attach a `unit_table` to `SpikeTrains` and select
  cells with `.filter(...)`; carve out running epochs with `restrict(...)`.
- **Decode as an object**: reach for `BayesianDecoder` when you want a
  `fit` / `predict` / `score` surface — remembering it is byte-exact with
  `decode_session`.
