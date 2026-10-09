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
env.units = "cm"
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
      f"{np.nanmedian(decoding_error(result.map_position, actual)):.1f} cm")

# --- BayesianDecoder: optional wrapper, byte-exact vs decode_session -------
decoder = BayesianDecoder(env, dt=0.1).fit(spike_times, times, positions)
prediction = decoder.predict(spike_times, times)
assert np.array_equal(prediction.posterior, result.posterior)  # byte-for-byte
error = decoder.score(spike_times, times, positions, metric="median_error")
print(f"BayesianDecoder.score median error: {error:.1f} cm")
```

## Simulator attributes feed analyses

A `SimulationSession` carries `env`, `spike_times`, `unit_ids`, `times`,
`positions`, `models`, integer-label `ground_truth` and `metadata`. Its field
bindings are frozen; spike arrays, labels and models must align one-to-one.
Unit labels also select the ground truth for validation and the rate panels
for plotting.

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
validation = validate_simulation(sim, unit_ids=[0, 2, 4])
```

`plot_session_summary(sim, unit_ids=[1, 3])` uses the same label selection.
Analysis functions take arrays; the validation and summary plot functions
need the simulator holder because it carries simulation-specific data.

## `SpikeTrains` and epoch selection

`SpikeTrains` is a frozen bundle of ragged per-unit spike trains plus their
identity labels (`unit_ids`) and an optional per-unit metadata table
(`unit_table`). It gives you label access, iteration, and a metadata-driven
`filter`, and it flows **directly** into the batch encoding / decoding functions
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
st[9]             # label access by unit id -> array([0.5, 3. , 6. ])

# Metadata-driven selection (returns a new SpikeTrains).
ca1 = st.filter("region == 'CA1' and quality > 0.5")

# Flows straight into batch compute — unit_ids are carried into the result.
rates = compute_spatial_rates(env, st, times, positions)
```

### Restricting to epochs

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

`epochs` accepts several forms: `(start, end)` scalars, an `(n, 2)` array,
parallel `(starts, ends)` 1-D arrays (whose length is **not** 2), or a pynapple
`IntervalSet` (duck-typed — no pynapple import).

!!! warning "The one ambiguous `epochs` form"
    A bare **length-2 pair of length-2 sequences** (e.g. `[[0, 5], [10, 15]]`)
    is ambiguous — it could mean two `(start, end)` interval rows *or* two
    parallel `(starts, ends)` arrays — so it **raises**. Disambiguate by passing
    an `(n, 2)` NumPy array (`np.asarray([[0, 5], [10, 15]])`) for interval rows,
    or explicit 1-D `start` / `end` arrays.

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

The functional `decode_session` remains the primary path — `predict` reproduces
it **byte-for-byte** on the same inputs and parameters, so the object wrapper is
purely for callers who prefer a `fit` / `predict` / `score` object.

Because decoding runs through the `Environment`, `BayesianDecoder` decodes
**linearized tracks**, masked open fields, and graph-based layouts — not just a
rectangular grid — and `score(..., distance="geodesic")` measures error along the
environment's connectivity graph. That is a differentiator over pynapple's
`decode_1d` / `decode_2d`.

## pynapple interop

pynapple objects convert to and from plain arrays **at the boundary** with
`from_pynapple` / `to_pynapple`, so the scientific code never touches pynapple.

!!! note "Requires the `pynapple` extra"
    `pip install neurospatial[pynapple]`. Only these two adapter functions
    import pynapple, and they import it lazily.

```python
from neurospatial.io import from_pynapple, to_pynapple

# Ingress: pynapple -> plain arrays.
trains, unit_ids = from_pynapple(tsgroup)      # TsGroup   -> (trains, unit_ids)
times, positions = from_pynapple(tsdframe)     # Tsd/TsdFrame -> (times, positions)
start, end = from_pynapple(intervalset)        # IntervalSet -> (start, end)

# Egress: a decoded MAP track -> a pynapple Tsd / TsdFrame.
tsd = to_pynapple(result)                       # from a DecodingResult
```

A raw `TsGroup` can supply spikes and unit labels directly. Pass tracking as
two explicit arrays, `tsdframe.t` and `tsdframe.values`, with timestamps first:

```python
from neurospatial.encoding import compute_spatial_rates
from neurospatial.decoding import decode_session

rates = compute_spatial_rates(env, tsgroup, tsdframe.t, tsdframe.values)
result = decode_session(env, tsgroup, tsdframe.t, tsdframe.values, dt=0.1)
```

neurospatial reads only the spike times and labels from a `TsGroup`; it
ignores the group's `time_support`. Without `spike_window=`, spikes are assumed
to have been recorded wherever position was tracked. When the electrophysiology
covers a different stretch, pass that coverage explicitly. An `IntervalSet` is
accepted directly:

```python
rates = compute_spatial_rates(
    env, tsgroup, tsdframe.t, tsdframe.values, spike_window=recording_intervals
)
```

Use `tsgroup.time_support` here only if you set it to the true acquisition
intervals: a `TsGroup` built from spike times alone gets a default support that
runs from its first spike to its last, which is not recording coverage.

## NWB interop

The NWB adapters read population spikes, position, and pose out of an NWB file,
and round-trip a `SpatialRatesResult` back into one.

!!! note "Requires the `nwb` extra"
    `pip install neurospatial[nwb]`. As with pynapple, `import neurospatial`
    never imports `pynwb`; the readers import it only when called.

Read components explicitly. These readers return frozen, non-iterable holders:

| Reader | Holder attributes |
| --- | --- |
| `read_position` | `NWBPosition.times`, `.positions`, `.units` |
| `read_head_direction` | `NWBHeadDirection.times`, `.headings` (radians) |
| `read_units` | `NWBUnits.spike_times`, `.unit_ids`, `.obs_intervals`, `.spike_window` |

Position values already include the stored conversion and offset. Recognized
unit aliases become `m`, `cm`, `mm` or `px` without scaling those values again.
Other nonempty declarations remain visible; `None` means no unit was declared.
Choose the physical unit explicitly in that case before creating an environment.
`environment_from_position` retains its warned cm fallback when used without
an explicit unit.

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
from neurospatial.io.nwb import read_intervals, read_position, read_units

with NWBHDF5IO("session.nwb", "r") as io:
    file = io.read()
    units = read_units(file)  # Or unit_ids=[7, 11] to select table labels.
    pos = read_position(file)
    epoch_table = read_intervals(file, "epochs")
    chosen = epoch_table["tags"].map(lambda tags: "run" in tags)
    epochs = epoch_table.loc[chosen, ["start_time", "stop_time"]].to_numpy(dtype=float)

# Eager arrays and interval metadata remain usable after the file closes.
position_units = pos.units
if position_units is None:
    raise ValueError("Choose position_units explicitly from the experiment's physical units.")
env = Environment.from_samples(pos.positions, bin_size=5.0, units=position_units)
rates = compute_spatial_rates(
    env, units.spike_times, pos.times, pos.positions, unit_ids=units.unit_ids,
    epochs=epochs, spike_window=units.spike_window, fill_value=0.0,
)
rate_table = rates.summary_table()
print(rate_table)

# Reuse the maps and their unit labels; train order is retained for these arrays.
decoder = BayesianDecoder.from_rates(rates, dt=0.2)
result = decoder.predict(
    units.spike_times, pos.times, epochs=epochs, spike_window=units.spike_window,
)
print(result.summary())

# result.times contains only observed-run bins; do not make a clock across gaps.
actual = np.column_stack([
    np.interp(result.times, pos.times, pos.positions[:, dim])
    for dim in range(env.n_dims)
])
ax = result.plot(show_map=True, colorbar=True)
plot_times = ax.lines[0].get_xdata()  # Seconds on continuous clocks, indices across gaps.
actual_line = ax.plot(plot_times, env.bin_at(actual), "c--", label="Actual spatial bin")[0]
ax.legend()
plt.show()
```

The posterior plot's y-axis is spatial-bin indices; physical positions belong
in separate position plots or accuracy metrics. The actual overlay shares the
MAP line's x coordinates, including the index axis used across gaps. See the
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
connected copy of the `Environment` — and `read_place_field` reconstructs an
equal result. Because the environment round-trips with its connectivity intact,
graph operations work on the restored env with no `env=` argument.

```python
from pynwb import NWBHDF5IO
from neurospatial.encoding import compute_spatial_rates
from neurospatial.io.nwb import write_spatial_rates, read_place_field

rates = compute_spatial_rates(env, spike_times, times, positions)

# Write.
with NWBHDF5IO("session.nwb", "r+") as io:
    nwbfile = io.read()
    write_spatial_rates(nwbfile, rates, name="ca1_place_fields")
    io.write(nwbfile)

# Read back (env restored from the file; pass env= to override).
with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()
    restored = read_place_field(nwbfile, name="ca1_place_fields")
    restored.firing_rates.shape   # (n_units, n_bins)
```

## See Also

- **[Complete Workflows](workflows.md)**: End-to-end encode / decode examples
- **[Spatial Analysis](spatial-analysis.md)**: Occupancy, fields, and trajectory operations
- **[API Reference](../api/index.md)**: NWB holders, `SpikeTrains`, `BayesianDecoder`, and the interop adapters
- **[Loading from NWB notebook](../examples/27_loading_from_nwb.ipynb)**: A worked NWB read example

## Next Steps

- **Read components**: select units and epochs explicitly, then pass holder
  attributes and acquisition coverage to your analysis.
- **Filter and restrict**: attach a `unit_table` to `SpikeTrains` and select
  cells with `.filter(...)`; carve out running epochs with `restrict(...)`.
- **Decode as an object**: reach for `BayesianDecoder` when you want a
  `fit` / `predict` / `score` surface — remembering it is byte-exact with
  `decode_session`.
