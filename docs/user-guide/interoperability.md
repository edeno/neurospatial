# Interoperability

neurospatial is **array-first**. Every analysis in the library runs on plain
NumPy arrays and needs **no optional dependencies** — you build an
[`Environment`](environments.md), pass spike times / timestamps / positions as
arrays, and get results back as arrays (or result objects that wrap them).

The pieces on this page — the pynapple and NWB adapters, the
[`Session`](#bundle-your-data-with-session) bundle,
[`SpikeTrains`](#spiketrains-and-epoch-selection), and the
[`BayesianDecoder`](#bayesiandecoder) object — are **optional conveniences**.
They form a *session-first* ergonomic layer that sits **on top of** the array
path, never replacing it. The compute functions are the same either way, so you
can move between the two freely.

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
The example first computes rate maps and decodes position **array-first**, then
shows the *same* analysis bundled with `Session` and driven through the
`BayesianDecoder` object — all composing with the **identical** compute
functions.

```python
import numpy as np

from neurospatial import Environment, Session
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
    cells, positions, times, seed=0, show_progress=False
)

# Rate maps and a one-call decode, straight from arrays.
rates = compute_spatial_rates(env, spike_times, times, positions)
result = decode_session(env, spike_times, times, positions, dt=0.1)
actual = np.interp(result.times, times, positions[:, 0]).reshape(-1, 1)
print(f"array-first median error: "
      f"{np.nanmedian(decoding_error(result.map_position, actual)):.1f} cm")

# --- Session bundle: the SAME arrays, grouped for discoverability ----------
sess = Session.from_arrays(
    env=env, times=times, positions=positions, spike_times=spike_times
)
# Accessors expose the raw arrays; compute stays functional (bundle, not
# god-object) — pass the bundle's fields straight to the same function.
rates_via_session = compute_spatial_rates(
    sess.env, sess.spikes, sess.times, sess.positions
)
assert np.array_equal(rates.firing_rates, rates_via_session.firing_rates)

# --- BayesianDecoder: optional wrapper, byte-exact vs decode_session -------
decoder = BayesianDecoder(env, dt=0.1).fit(spike_times, times, positions)
prediction = decoder.predict(spike_times, times)
assert np.array_equal(prediction.posterior, result.posterior)  # byte-for-byte
error = decoder.score(spike_times, times, positions, metric="median_error")
print(f"BayesianDecoder.score median error: {error:.1f} cm")
```

The array-first and session-first paths are not alternatives to choose between —
they *compose*. Bundle your data with `Session` for discoverability, then hand
`sess.env` / `sess.spikes` / `sess.times` / `sess.positions` to any function in
the library.

## Bundle your data with `Session`

A `Session` groups the objects an analysis revolves around — the spatial `env`,
the animal's position (`times` + `positions`), the population `spikes` (with
unit identity), optional `epochs`, and free-form `metadata` — into a single
immutable, discoverable bundle.

```python
from neurospatial import Session

sess = Session.from_arrays(
    env=env,
    times=times,
    positions=positions,
    spike_times=spike_times,
    unit_ids=None,        # defaults to np.arange(n_units)
    unit_table=None,      # optional per-unit metadata DataFrame
    metadata={"subject": "rat042", "session": "run1"},
)

# Accessors expose the raw arrays (and a SpikeTrains).
sess.times       # (n_samples,) timestamps
sess.positions   # (n_samples, n_dims) coordinates
sess.env         # the Environment (or None)
sess.spikes      # a SpikeTrains carrying unit_ids / unit_table
```

`Session` is a **discoverability bundle, not a god-object**: it carries data and
exposes the raw arrays, but holds **no** heavy analysis methods. Compute stays
functional — you pass the bundle's fields to the same free functions:

```python
from neurospatial.encoding import compute_spatial_rates

rates = compute_spatial_rates(
    sess.env, sess.spikes, sess.times, sess.positions
)
```

The bundle is **frozen**. The two "modifiers" return a **new** `Session` and
never mutate the original:

```python
# Attach or swap the environment (returns a new Session).
sess = sess.with_environment(env)

# Restrict to epochs (returns a new Session). The spike restriction is
# identity-preserving: it trims spikes per unit but never drops units, so
# unit_ids and unit_table ride along unchanged.
run_epochs = np.array([[0.0, 30.0], [60.0, 90.0]])
run = sess.restrict(run_epochs)
```

To load a session straight from an NWB file, use `load_session` (see
[NWB interop](#nwb-interop)):

```python
from neurospatial import load_session

sess = load_session("session.nwb")   # requires the `nwb` extra
```

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
from neurospatial import SpikeTrains

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
boolean mask; and `restrict_spike_trains(trains, epochs)` masks *ragged* trains
(each unit by its own timestamps).

```python
from neurospatial import restrict
from neurospatial.behavior import in_epochs, restrict_spike_trains

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
fitted = decoder.fit(spike_times, times, positions, epoch=(0.0, 60.0))
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

You often do not even need the adapter: a raw `TsGroup` (spikes) or `Tsd` /
`TsdFrame` (position) flows **directly** into the compute functions, which accept
the pynapple-group and position-source surfaces:

```python
from neurospatial.encoding import compute_spatial_rates
from neurospatial.decoding import decode_session

# Pass pynapple objects straight through — no manual conversion.
rates = compute_spatial_rates(env, tsgroup, tsdframe)
result = decode_session(env, tsgroup, tsdframe, dt=0.1)
```

## NWB interop

The NWB adapters read population spikes, position, and pose out of an NWB file,
and round-trip a `SpatialRatesResult` back into one.

!!! note "Requires the `nwb` extra"
    `pip install neurospatial[nwb]`. As with pynapple, `import neurospatial`
    never imports `pynwb`; the readers import it only when called.

The quickest entry point is `Session.from_nwb` / `load_session`, which reads the
units, position, and (if present) a persisted environment into a `Session`:

```python
from neurospatial import Session, load_session

sess = Session.from_nwb("session.nwb")
sess = load_session("session.nwb")            # dispatches to Session.from_nwb
```

For finer control, read individual components. `read_units` returns
`(trains, unit_ids)` — the standard spike input for the batch functions:

```python
from pynwb import NWBHDF5IO
from neurospatial.io.nwb import read_units, read_position, read_pose

with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()
    trains, unit_ids = read_units(nwbfile)
    positions, timestamps = read_position(nwbfile)
```

`read_units`, `read_position`, and `read_pose` accept `lazy=True`, which returns
handles that materialize their data only when sliced or `np.asarray`-ed — useful
for large recordings.

!!! warning "Lazy handles are only valid while the file is open"
    A `lazy=True` handle reads from the open `NWBFile` / `NWBHDF5IO`. Materialize
    it (index it or `np.asarray` it) **inside** the `with NWBHDF5IO(...)` block;
    `lazy=False` (the default) returns arrays that stay valid after the file
    closes.

```python
with NWBHDF5IO("session.nwb", "r") as io:
    nwbfile = io.read()
    trains, unit_ids = read_units(nwbfile, lazy=True)
    first_unit = np.asarray(trains[0])   # materialize WHILE the file is open
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
- **[API Reference](../api/index.md)**: `Session`, `SpikeTrains`, `BayesianDecoder`, and the interop adapters
- **[Loading from NWB notebook](../examples/27_loading_from_nwb.ipynb)**: A worked NWB read example

## Next Steps

- **Bundle a session**: wrap your arrays in `Session.from_arrays(...)` and pass
  its fields to the compute functions you already use.
- **Filter and restrict**: attach a `unit_table` to `SpikeTrains` and select
  cells with `.filter(...)`; carve out running epochs with `restrict(...)`.
- **Decode as an object**: reach for `BayesianDecoder` when you want a
  `fit` / `predict` / `score` surface — remembering it is byte-exact with
  `decode_session`.
</content>
</invoke>
