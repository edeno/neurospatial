# Researcher-workflow checkpoint — 2026-10-07

**Decision: HOLD before Phase 5a.** The array workflows are easier to compose, and
all five journeys execute using public documentation and public runtime help.
The track journey still receives a painful rating, and the checkpoint exposed a
simulation-duration bug and scientific interpretation errors. A corrective
Phase 4c must merge and this checkpoint must be repeated before Phase 5a starts.
This review does not implement those fixes or mark the checkpoint passed.

Reviewed integration commit `276c8e2be32d0b2ca6327ef89add878397e819a7`, after
Phase 4b merged as `b908d1da` (PR #45). See the
[UX review](UX_REVIEW_2026-10-07.md) and
[design review](DESIGN_REVIEW_2026-10-07.md).

## Method and limits

Two independent reviewers attempted the journeys defined in
`.claude/workflows/design-review.js`; the lead also ran those paths and the
three probes from `.claude/workflows/ux-review.js`. The Claude-specific runner
was unavailable, so the workflow scopes, rating vocabulary and report formats
were adapted to direct execution and the two authorized reviewers. The
checkpoint's public-documentation restriction took precedence over the
scripts' instructions to inspect implementation code.

Call discovery used README, public guides, public notebook Python companions,
public exports, `inspect.signature`, `inspect.getdoc`, and result attributes.
**Implementation-file reads: 0. Test-file reads: 0. Required source-reading
fallbacks: 0**, for the lead and both reviewers. Automatically printed traceback
filenames and warning locations were observed without opening implementation
files. Plans were read only for scope and to assign follow-up tasks.

Scripts ran through `uv run` from temporary working directories with Matplotlib
Agg and all integration extras installed. Versions: neurospatial 0.8.0,
Python 3.13, NumPy 2.3.4, pynwb 3.1.2, Matplotlib 3.10.7, and
track-linearization 2.4.0. No interactive browser, Napari or widget session was
run. HTML accessibility observations concern generated markup/images, not
screen-reader certification. No held-out decoding accuracy or biological
classification accuracy is established by these synthetic examples.

The available October baseline is the plan's qualitative record: UX
**CONFUSING**, all five design journeys **painful**. The original session-only
reports and their numerical call/line counts are unavailable. Consequently,
this review reports current counts and qualitative changes; it claims no
percentage reduction in code or calls.

## Journey measurements

LOC counts nonblank, noncomment physical lines, including imports, fixtures,
plots, reporting and checks. Calls count syntactic AST `Call` sites, including
NumPy, pandas and printing; they are not loop invocation counts. The separately
listed neurospatial calls include documented model/result/environment methods.
These are whole executed scripts, not normalized minimum-code estimates.

| Journey | Current rating | LOC | All calls | neurospatial calls | Source reads |
| --- | --- | ---: | ---: | ---: | ---: |
| NWB components → epochs → population fields → decode → table/plot/animation | workable | 25 | 16 | 10 | 0 |
| Linear track → graph linearization → trials → direction-specific fields | painful | 42 | 54 | 12 | 0 |
| Heading/egocentric frame → object-vector map and candidate screen | workable | 33 | 59 | 24 | 0 |
| Simulation → population encoding/decoding → assembly/reactivation statistics | workable via open-field helper; painful via linear helper | 26 | 53 | 13 | 0 |
| Events → PSTH/raster → GLM regressors → positions | smooth on continuous data; workable across gaps | 34 | 56 | 9 | 0 |

The independently attempted NWB guide route used `load_session` and
`Session.restrict`: 16 LOC, 23 calls, 8 neurospatial calls. It succeeded after
replacing `encoding_models=rates` with `rates.firing_rates`, but retained the
soon-to-be-removed Session dependency. The primary measurement above uses the
lead's component-reader/raw-array route instead. Fixture creation for the NWB
input is listed separately in the reproduction appendix.

### What completed

- **NWB:** real HDF5 input, 6 units with IDs 100–105, 6,000 2D samples,
  selected epochs `[0,20)` and `[35,55)`, 40 seconds shared occupancy. Unit IDs
  survived into the summary table. The lead exported a real field PNG and
  posterior/position-overlay HTML. The independent decode had shape `(160,78)`
  at `dt=0.25`; row sums were within floating-point tolerance of one, with
  zero bins in the excluded 20–35 second interval.
- **Track:** 20 graph bins, 6 successful outbound and 6 successful inbound
  trials. Two labeled maps peaked at 57.5 and 42.5 cm near their planted
  direction-specific centers. Region/trial labels, rather than linearization,
  supplied direction separation.
- **Object-vector:** 15,000 samples, valid headings, bearing/distance arrays
  `(15000,1)`, an egocentric polar plot, and a Boolean candidate screen.
  Preferred distance was 14 cm for 15 cm truth; direction was 75° for 90°
  truth. A place-cell control also passed the information-only screen, an
  already documented limitation owned by Phase 5b.
- **Population statistics:** the open-field helper produced 6,000 samples,
  59.99 seconds, 12-unit maps `(12,169)`, counts `(599,12)` and posterior
  `(599,169)`. Manual and one-call decoders matched exactly. Spike counts fed
  assembly/EV analyses directly; these are a sibling branch to decoding, not
  analyses of posterior probabilities. An automatically selected PCA pattern
  had no core members, illustrating why a significant dimension alone does
  not establish a biological assembly. In-sample median error, 7.563 cm, is
  a composition check rather than held-out performance.
- **Events:** 203 spikes, 11 retained events, an 11-row raster, a `(1200,6)`
  regressor matrix, and valid event-position/bin assignments. The peak PSTH
  was 43.636 Hz at +0.2125 seconds. SEM was explicitly converted from counts
  to Hz in the manual table; the plot helper handled that conversion.

## Confirmed findings and disposition

| Finding | Evidence | Action |
| --- | --- | --- |
| Linear-track duration does not control returned time | Requests 1 and 2 seconds both yield 9,387 samples spanning 18.772 seconds; metadata records the requested, different duration. | New Phase 4c.1; checkpoint blocker. |
| Linearization tutorial promises direction-specific coordinates | `[25,50,75,50,25]` maps to the same coordinates and bins `[4,9,14,9,4]`, contradicting the monotonic/direction-separation explanation. | New Phase 4c.2; checkpoint blocker. |
| Statistical effect/activation thresholds are described as significance | A documented one-neuron assembly with 90 zeros/10 ones yields standardized values `[-1/3,3]`: 10% exceed 2. Std remains 1; normalization is correct. EV result docs call EV >0.1 significant and misdescribe controlled REV. | New Phase 4c.3; correct interpretation without changing calculations. |
| Native 1D grid rate plot fails | `rates.plot(idx=0)` raises `NotImplementedError: pcolormesh requires 2D grids, got grid_shape=(7,)`. Graph-track plotting works. | Phase 7 checkpoint addition: support native 1D plots; public Matplotlib line-plot recovery exists. |
| Gap-filtered event cohorts do not compose automatically | PSTH retains 2/drops 2 events; raster returns 4 rows; positioned-event table returns 4 rows, 2 with NaN coordinates. No retained-event mask/IDs are returned. | Phase 6b checkpoint addition: an executable common-cohort recipe and explicit closure guidance, without changing deferred helpers' numerical semantics. |
| NWB entry path/units/epochs require joining recipes | Guide favors Session; fixture's loader has `epochs=None`. Component readers and interval-table extraction work. | Phase 6c checkpoint addition: preserve/update the joined raw-array recipe when holders replace tuples and Session is removed. |
| Population statistics lack a joined public example | Exported assembly/EV functions were found through runtime docs, not a public workflow example/API-index entry. Zero-variance units require consistent cross-period filtering. | Phase 6a checkpoint addition: public navigation and a runnable count-based branch. |
| Candidate OVC screen accepts a place-cell control | Control information 1.0982 bits/spike, default candidate verdict True; tutorial already acknowledges the limitation. | Existing Phase 5b; add explicit estimator/bias guidance, no threshold change. |
| Summary output differs between single/population | Single table has 4 columns, population 9; scalar repr aggregation is unnamed, table attrs empty, singular xarray absent. | Existing Phase 7.1–7.3; preserve its current scope and validations. |
| HTML colorbars are unavailable | `show_colorbar=False/True` yields byte-identical first-frame PNGs; runtime help explicitly says the options are not implemented. | Optional output polish; document the limitation. Static plots already have viridis and labeled numeric colorbars. |

Small caller mistakes were recoverable from public help: passing a result object
to `decode_session(encoding_models=...)`, assuming `from_graph(units=...)`, and
omitting headings from `compute_egocentric_distance`. These are recorded
attempts, not unsupported claims that the documentation promises those forms.
The session decoder's Notes nevertheless say “pass the result” despite its
ndarray parameter; Phase 6b's split must replace that ambiguous guidance.

PSTH uses half-open windows, while raster alignment includes the stop spike and
count/indicator regressors use inclusive boundaries. Adjacent regressor windows
both count a shared boundary event. Do not “fix” these existing contracts by
forcing a universal closure. Document and compose them deliberately.

## Gate and next step

Overall UX improves from the available CONFUSING baseline to **NEEDS_POLISH**.
That improvement is not an unqualified pass: journey 2 remains painful, and
simulation metadata and interpretation claims can mislead scientific work.
Phase 4c contains the bounded corrections needed before repeating the gate.
Medium-priority composition/output work is attached to the already planned
Phases 5b, 6a–6c and 7. No Phase 5a implementation is started by this review.

## Reproduction appendix

The following code is the exact executed checkpoint evidence. Outputs belong
in a temporary directory; set `MPLBACKEND=Agg` and use `uv run --project` pointing
at the checkout. Scripts include reporting/fixture code, which is why their
counts must not be interpreted as minimum user ceremony. HTML exports can
warn about playback-speed caps after explicit frame subsampling.


### NWB input fixture (not counted as journey 1)

Exact script: `fixture.py`; 15 physical LOC, 14 call sites.

```python
from datetime import datetime, timezone

from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import Position, SpatialSeries
from neurospatial.simulation import open_field_session

session = open_field_session(duration=60.0, arena_size=60.0, bin_size=5.0, n_place_cells=6, seed=7)
nwb = NWBFile("Checkpoint synthetic input", "checkpoint", datetime(2026, 10, 7, tzinfo=timezone.utc))
position = Position(name="Position")
position.add_spatial_series(SpatialSeries(name="xy", data=session.positions, timestamps=session.times, reference_frame="arena origin", unit="cm"))
nwb.create_processing_module("behavior", "Tracking").add(position)
for i, spikes in enumerate(session.spike_trains):
    nwb.add_unit(id=100 + i, spike_times=spikes)
nwb.add_epoch(start_time=0.0, stop_time=20.0, tags=["run"])
nwb.add_epoch(start_time=35.0, stop_time=55.0, tags=["run"])
with NWBHDF5IO("session.nwb", "w") as io:
    io.write(nwb)
```

### Journey 1 — component readers and arrays

Exact script: `journey1.py`; 25 physical LOC, 16 call sites.

```python
import numpy as np
from pynwb import NWBHDF5IO
from neurospatial import Environment
from neurospatial.io.nwb import read_position, read_units, read_intervals
from neurospatial.encoding import compute_spatial_rates
from neurospatial.decoding import decode_session
from neurospatial.animation import PositionOverlay

with NWBHDF5IO("session.nwb", "r") as io:
    nwb = io.read()
    positions, times = read_position(nwb)
    spikes, unit_ids = read_units(nwb)
    intervals = read_intervals(nwb, "epochs")
epochs = intervals[["start_time", "stop_time"]].to_numpy()
env = Environment.from_samples(positions, bin_size=5.0, units="cm")
rates = compute_spatial_rates(env, spikes, times, positions, epochs=epochs,
                              unit_ids=unit_ids, fill_value=0.0)
decoded = decode_session(env, spikes, times, positions, epochs=epochs,
                         encoding_models=rates.firing_rates, dt=0.1)
table = rates.summary_table(include_classification=False)
ax = rates.plot(idx=0)
ax.figure.savefig("nwb-field.png")
overlay = PositionOverlay(positions=positions, times=times)
animation = env.animate_fields(decoded.posterior[::20], frame_times=decoded.times[::20],
                               overlays=[overlay], backend="html", save_path="posterior.html")
print(table.to_string())
```

### Journey 2 — graph geometry and explicit direction labels

Exact script: `track_workflow.py`; 42 physical LOC, 54 call sites.

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from neurospatial import Environment
from neurospatial.behavior import segment_trials, goal_pair_direction_labels
from neurospatial.encoding import compute_directional_place_fields

graph = nx.Graph()
graph.add_node(0, pos=(0.0, 0.0))
graph.add_node(1, pos=(100.0, 0.0))
graph.add_edge(0, 1, edge_id=0, distance=100.0)
env = Environment.from_graph(graph, edge_order=[(0, 1)], edge_spacing=0.0, bin_size=5.0, name='Linear track')
env.units = 'cm'
times = np.arange(0.0, 60.0, 0.05)
phase = (times % 10.0) / 10.0
x = 10.0 + 80.0 * (1.0 - np.abs(2.0 * phase - 1.0))
positions = np.column_stack([x, np.zeros_like(x)])
linear = env.to_linear(positions)
print('track:', env.n_bins, 'bin_centers:', env.bin_centers.shape, 'linear shape:', linear.shape)
opposite = np.array([[25.0, 0.0], [50.0, 0.0], [75.0, 0.0], [50.0, 0.0], [25.0, 0.0]])
print('opposite traversal linear:', env.to_linear(opposite).tolist(), 'bins:', env.bin_at(opposite).tolist())
env.regions.add('home', polygon=Polygon([(-1, -5), (15, -5), (15, 5), (-1, 5)]))
env.regions.add('goal', polygon=Polygon([(85, -5), (101, -5), (101, 5), (85, 5)]))
position_bins = env.bin_sequence(times, positions, dedup=False)
outbound = segment_trials(position_bins, times, env, start_region='home', end_regions=['goal'])
inbound = segment_trials(position_bins, times, env, start_region='goal', end_regions=['home'])
labels = goal_pair_direction_labels(times, outbound + inbound)
rng = np.random.default_rng(7)
peak = np.where(phase < 0.5, 60.0, 40.0)
intensity = 0.5 + 25.0 * np.exp(-0.5 * ((x - peak) / 10.0) ** 2)
spikes = times[rng.random(len(times)) < intensity * 0.05]
fields = compute_directional_place_fields(env, spikes, times, positions, labels)
print('trials:', len(outbound), len(inbound), 'successful:', sum(t.success for t in outbound + inbound))
print('labels:', dict(zip(*np.unique(labels, return_counts=True))))
print('fields:', fields.labels, 'occupancy seconds:', {label: round(fields.occupancy[label].sum(), 3) for label in fields.labels})
fig, axes = plt.subplots(1, len(fields.labels), figsize=(10, 3), constrained_layout=True)
for ax, label in zip(axes, fields.labels, strict=True):
    env.plot_field(fields.firing_rates[label], ax=ax, colorbar_label='Firing rate (Hz)')
    ax.set_title(label)
    print('peak:', label, env.bin_centers[np.nanargmax(fields.firing_rates[label])].tolist(), round(np.nanmax(fields.firing_rates[label]), 3))
fig.savefig('directional_fields.png')
plt.close(fig)
```

### Journey 3 — egocentric map and candidate screen

Exact script: `journey3_egocentric.py`; 33 physical LOC, 59 call sites.

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from neurospatial import Environment
from neurospatial.encoding import compute_egocentric_rate, is_object_vector_cell, object_vector_score
from neurospatial.ops.egocentric import EgocentricFrame, heading_from_velocity, compute_egocentric_bearing, compute_egocentric_distance
from neurospatial.simulation import ObjectVectorCellModel, PlaceCellModel, generate_poisson_spikes, simulate_trajectory_ou

xx, yy = np.meshgrid(np.linspace(0, 60, 31), np.linspace(0, 60, 31))
env = Environment.from_samples(np.column_stack([xx.ravel(), yy.ravel()]), bin_size=3.0)
env.units = "cm"
positions, times = simulate_trajectory_ou(env, duration=300.0, dt=0.02, speed_units="cm", seed=42)
headings = heading_from_velocity(positions, times, min_speed=2.0, bandwidth=3.0)
objects = np.array([[30.0, 30.0]])
bearings = compute_egocentric_bearing(positions, headings, objects)
distances = compute_egocentric_distance(positions, headings, objects)
frame = EgocentricFrame(position=np.array([0.0, 0.0]), heading=np.pi / 2)
frame_example = frame.to_egocentric(np.array([[10.0, 0.0]]))
ovc = ObjectVectorCellModel(env, object_positions=objects, preferred_distance=15.0, distance_width=4.0, preferred_direction=np.pi/2, max_rate=60.0, baseline_rate=0.05)
place = PlaceCellModel(env, center=np.array([20.0, 20.0]), width=6.0, max_rate=40.0, baseline_rate=0.1)
spikes = [generate_poisson_spikes(ovc.firing_rate(positions, headings=headings), times, seed=42), generate_poisson_spikes(place.firing_rate(positions), times, seed=43)]
rows = []
for name, train in zip(["object_vector", "place_control"], spikes, strict=True):
    result = compute_egocentric_rate(env, train, times, positions, headings, objects, distance_range=(0.0, 40.0), n_distance_bins=10, n_direction_bins=12, method="gaussian_kde", bandwidth=1.0, min_occupancy=0.1)
    raw = compute_egocentric_rate(env, train, times, positions, headings, objects, distance_range=(0.0, 40.0), n_distance_bins=10, n_direction_bins=12)
    score = object_vector_score(np.asarray(result.firing_rate).reshape(result.n_distance_bins, result.n_direction_bins))
    quick = is_object_vector_cell(env, train, times, positions, headings, objects, distance_range=(0.0, 40.0), n_distance_bins=10, n_direction_bins=12)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    result.plot(ax=ax)
    fig.savefig(f"journey3_{name}.png")
    plt.close(fig)
    rows.append({"cell": name, "spikes": len(train), "preferred_distance_cm": result.preferred_distance(), "preferred_direction_degrees": float(np.degrees(result.preferred_direction())), "score": score, "smoothed_info_bits_per_spike": result.egocentric_spatial_information(), "raw_info_bits_per_spike": raw.egocentric_spatial_information(), "quick_classification": bool(quick), "raw_result_classification": bool(raw.is_object_vector_cell()), "smoothed_result_classification": bool(result.is_object_vector_cell()), "occupancy_seconds": float(np.sum(result.occupancy)), "spike_window_assumed": bool(result.spike_window_assumed)})
summary = {"trajectory_shape": list(positions.shape), "heading_finite": int(np.isfinite(headings).sum()), "bearing_shape": list(bearings.shape), "distance_shape": list(distances.shape), "facing_north_east_target_egocentric": frame_example.tolist(), "rows": rows}
print(json.dumps(summary, indent=2))
```

### Journey 4 — open-field recovery and count-based statistics

Exact script: `journey4_population.py`; 26 physical LOC, 53 call sites.

```python
import json
import numpy as np
from neurospatial.simulation import open_field_session
from neurospatial.encoding import compute_spatial_rates
from neurospatial.decoding import bin_spikes_in_time, decode_position, decode_session, decoding_error, detect_assemblies, assembly_activation, pairwise_correlations, explained_variance_reactivation, reactivation_strength

session = open_field_session(duration=60.0, arena_size=60.0, bin_size=5.0, n_place_cells=12, seed=42)
env, trains, times, positions = session.env, session.spike_trains, session.times, session.positions
rates = compute_spatial_rates(env, trains, times, positions, bandwidth=5.0, min_occupancy=0.1, fill_value=0.0)
counts, centers = bin_spikes_in_time(trains, 0.1, t_start=float(times[0]), t_stop=float(times[-1]))
decoded = decode_position(env, counts, rates.firing_rates, 0.1, times=centers)
golden = decode_session(env, trains, times, positions, dt=0.1, bandwidth=5.0, min_occupancy=0.1)
actual = np.column_stack([np.interp(centers, times, positions[:, i]) for i in range(env.n_dims)])
errors = decoding_error(decoded.map_position, actual, env=env)
split = len(counts) // 3
pre, template, match = counts[:split], counts[split:2*split], counts[2*split:]
assemblies = detect_assemblies(template, algorithm="pca", rng=42)
correlations = [pairwise_correlations(period) for period in [pre, template, match]]
ev = explained_variance_reactivation(correlations[1], correlations[2], control_correlations=correlations[0])
ev_no_control = explained_variance_reactivation(correlations[1], correlations[2])
activation_rows = []
for pattern in assemblies.patterns:
    activation = assembly_activation(template, pattern)
    activation_rows.append({"members": pattern.member_indices.tolist(), "weight_norm": float(np.linalg.norm(pattern.weights)), "activation_mean": float(np.mean(activation)), "activation_std": float(np.std(activation)), "fraction_activation_above_2": float(np.mean(activation > 2)), "reactivation_strength": reactivation_strength(template, match, pattern), "stronger_match_strength": reactivation_strength(template, 3*template, pattern)})
summary = {"position_shape": list(positions.shape), "rates_shape": list(rates.firing_rates.shape), "counts_shape": list(counts.shape), "posterior_shape": list(decoded.posterior.shape), "posterior_row_sum_max_error": float(np.max(np.abs(decoded.posterior.sum(axis=1)-1))), "golden_vs_manual_max_error": float(np.max(np.abs(golden.posterior-decoded.posterior))), "in_sample_median_error_cm": float(np.median(errors)), "n_significant": assemblies.n_significant, "n_patterns": len(assemblies.patterns), "assembly_activations_shape": list(assemblies.activations.shape), "assembly_eigenvalues": assemblies.eigenvalues.tolist(), "assembly_threshold": float(assemblies.threshold), "activation_rows": activation_rows, "n_pairs": ev.n_pairs, "explained_variance": ev.explained_variance, "reversed_ev": ev.reversed_ev, "no_control_ev": ev_no_control.explained_variance, "no_control_rev": ev_no_control.reversed_ev}
print(json.dumps(summary, indent=2))
print(json.dumps({"requested_duration_seconds": 60.0, "actual_duration_seconds": float(times[-1]-times[0]), "total_spikes": sum(len(train) for train in trains)}))
```

### Journey 5 — PSTH, regressors and positioned events

Exact script: `journey5_events.py`; 34 physical LOC, 56 call sites.

```python
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neurospatial import Environment
from neurospatial.events import peri_event_histogram, align_spikes_to_events, time_to_nearest_event, event_count_in_window, event_indicator, add_positions, plot_peri_event_histogram

rng = np.random.default_rng(42)
times = np.arange(0.0, 60.0, 0.001)
event_times = np.arange(5.0, 60.0, 5.0)
firing_rate = np.full_like(times, 2.0)
for event in event_times:
    firing_rate += 30.0*np.exp(-0.5*((times-event-0.15)/0.08)**2)
spikes = times[rng.random(len(times)) < firing_rate*0.001]
result = peri_event_histogram(spikes, event_times, window=(-0.5, 1.0), bin_size=0.025, spike_window=(0.0, 60.0))
aligned = align_spikes_to_events(spikes, event_times, window=(-0.5, 1.0))
sample_times = np.arange(0.0, 60.0, 0.05)
regressor = time_to_nearest_event(sample_times, event_times, signed=True, max_time=2.0)
counts = event_count_in_window(sample_times, event_times, window=(-1.0, 0.0))
indicator = event_indicator(sample_times, event_times, window=(-1.0, 0.0))
positions = np.column_stack([30.0+20.0*np.cos(sample_times/3.0), 30.0+20.0*np.sin(sample_times/3.0)])
env = Environment.from_samples(positions, bin_size=3.0)
env.units = "cm"
events = add_positions(pd.DataFrame({"timestamp": event_times, "event_type": "reward"}), times=sample_times, positions=positions)
events["bin_index"] = env.bin_at(events[["x", "y"]].to_numpy())
trial_counts = np.array([np.sum((trial >= 0.05) & (trial < 0.35)) for trial in aligned])
events["response_spikes"] = trial_counts
design = np.column_stack([np.ones(len(sample_times)), regressor, counts, indicator.astype(float), positions])
fig, ax = plt.subplots()
plot_peri_event_histogram(result, ax=ax)
fig.savefig("journey5_psth.png")
plt.close(fig)
events.to_csv("journey5_event_positions.csv", index=False)
summary = {"spikes": len(spikes), "events_retained": result.n_events, "events_dropped": result.n_events_dropped, "peak_time_seconds": float(result.bin_centers[np.argmax(result.firing_rate)]), "peak_rate_hz": float(np.max(result.firing_rate)), "sem_count_to_rate_scale": 1/result.bin_size, "aligned_trials": len(aligned), "regressor_range_seconds": [float(np.min(regressor)), float(np.max(regressor))], "regressor_design_shape": list(design.shape), "indicator_matches_counts": bool(np.all(indicator == (counts > 0))), "event_bins": events.bin_index.tolist(), "event_position_missing": int(events[["x", "y"]].isna().any(axis=1).sum()), "response_count_mean": float(trial_counts.mean()), "event_rows": events.head(3).to_dict(orient="records")}
print(json.dumps(summary, indent=2))
```

### Small duration, 1D-plot and activation-interpretation probes

Exact script: `probe_time_and_statistics.py`; 21 physical LOC, 26 call sites.

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from neurospatial.simulation import linear_track_session
from neurospatial.encoding import compute_spatial_rates
from neurospatial.decoding import AssemblyPattern, assembly_activation

rows = []
for duration in [1.0, 2.0]:
    session = linear_track_session(duration=duration, track_length=1.0, bin_size=0.2, n_place_cells=2, n_laps=2, seed=42)
    rows.append({"requested_duration": duration, "observed_duration": float(session.times[-1]-session.times[0]), "samples": len(session.times), "units": session.env.units, "metadata": session.metadata})
rates = compute_spatial_rates(session.env, session.spike_trains, session.times, session.positions)
try:
    rates.plot(idx=0)
    plot_outcome = "success"
except Exception as exc:
    plot_outcome = f"{type(exc).__name__}: {exc}"
plt.close("all")
pattern = AssemblyPattern(weights=np.array([1.0]), member_indices=np.array([0]), explained_variance_ratio=1.0)
counts = np.concatenate([np.zeros(90), np.ones(10)]).reshape(-1, 1)
activation = assembly_activation(counts, pattern)
print(json.dumps({"linear_track_duration_probes": rows, "rates_plot_outcome": plot_outcome, "activation_mean": float(np.mean(activation)), "activation_std": float(np.std(activation)), "activation_values": np.unique(activation).tolist(), "fraction_above_2": float(np.mean(activation > 2))}, indent=2))
```

### Gap cohorts and endpoint probes

Exact script: `probe_event_windows.py`; 21 physical LOC, 37 call sites.

```python
import json
import warnings
import numpy as np
import pandas as pd
from neurospatial.events import peri_event_histogram, align_spikes_to_events, event_count_in_window, event_indicator, time_to_nearest_event, add_positions

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    edges = peri_event_histogram(np.array([0.0, 0.5, 1.0]), np.array([0.0]), window=(0.0, 1.0), bin_size=0.5)
edge_raster = align_spikes_to_events(np.array([0.0, 0.5, 1.0]), np.array([0.0]), window=(0.0, 1.0))
adjacent_counts = event_count_in_window(np.array([0.0, 0.5]), np.array([0.5]), window=(0.0, 0.5))
adjacent_indicators = event_indicator(np.array([0.0, 0.5]), np.array([0.5]), window=(0.0, 0.5))
event_times = np.array([0.1, 1.0, 2.1, 3.0])
spikes = np.array([0.1, 1.0, 2.1, 3.0])
recording = np.array([[0.0, 0.2], [2.0, 2.2]])
psth = peri_event_histogram(spikes, event_times, window=(-0.05, 0.05), bin_size=0.05, spike_window=recording)
rasters = align_spikes_to_events(spikes, event_times, window=(-0.05, 0.05))
times = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])
positions = np.column_stack([times, times])
events = add_positions(pd.DataFrame({"timestamp": event_times}), times=times, positions=positions)
regressors = time_to_nearest_event(np.array([0.09, 0.1, 0.11, 1.1, 2.1]), event_times)
print(json.dumps({"psth_edge_counts": edges.histogram.tolist(), "raster_edge_spikes": edge_raster[0].tolist(), "edge_warnings": [str(warning.message) for warning in caught], "inclusive_adjacent_count_bins": adjacent_counts.tolist(), "inclusive_adjacent_indicator_bins": adjacent_indicators.tolist(), "psth_retained_events": psth.n_events, "psth_dropped_events": psth.n_events_dropped, "raster_event_rows": len(rasters), "positioned_event_rows": events.to_dict(orient="records"), "regressor_signed_values": regressors.tolist(), "psth_public_attributes": [name for name in dir(psth) if not name.startswith('_')]}, indent=2))
```

### Exact small probe outputs

`probe_time_and_statistics.out`:

```json
{
  "linear_track_duration_probes": [
    {
      "requested_duration": 1.0,
      "observed_duration": 18.772000000000002,
      "samples": 9387,
      "units": "cm",
      "metadata": {
        "duration": 1.0,
        "n_cells": 2,
        "cell_type": "place",
        "trajectory_method": "laps",
        "coverage": "uniform",
        "seed": 42
      }
    },
    {
      "requested_duration": 2.0,
      "observed_duration": 18.772000000000002,
      "samples": 9387,
      "units": "cm",
      "metadata": {
        "duration": 2.0,
        "n_cells": 2,
        "cell_type": "place",
        "trajectory_method": "laps",
        "coverage": "uniform",
        "seed": 42
      }
    }
  ],
  "rates_plot_outcome": "NotImplementedError: pcolormesh requires 2D grids, got grid_shape=(7,)",
  "activation_mean": -1.7763568394002505e-17,
  "activation_std": 1.0,
  "activation_values": [
    -0.3333333333333333,
    3.0
  ],
  "fraction_above_2": 0.1
}
```

`probe_event_windows.out`:

```json
{
  "psth_edge_counts": [
    1.0,
    1.0
  ],
  "raster_edge_spikes": [
    0.0,
    0.5,
    1.0
  ],
  "edge_warnings": [
    "Computing PSTH with single event - SEM is undefined (will be NaN)."
  ],
  "inclusive_adjacent_count_bins": [
    1,
    1
  ],
  "inclusive_adjacent_indicator_bins": [
    true,
    true
  ],
  "psth_retained_events": 2,
  "psth_dropped_events": 2,
  "raster_event_rows": 4,
  "positioned_event_rows": [
    {
      "timestamp": 0.1,
      "x": 0.1,
      "y": 0.1
    },
    {
      "timestamp": 1.0,
      "x": NaN,
      "y": NaN
    },
    {
      "timestamp": 2.1,
      "x": 2.1,
      "y": 2.1
    },
    {
      "timestamp": 3.0,
      "x": NaN,
      "y": NaN
    }
  ],
  "regressor_signed_values": [
    -0.010000000000000009,
    0.0,
    0.009999999999999995,
    0.10000000000000009,
    0.0
  ],
  "psth_public_attributes": [
    "bin_centers",
    "bin_size",
    "firing_rate",
    "histogram",
    "n_events",
    "n_events_dropped",
    "plot",
    "sem",
    "summary",
    "to_dataframe",
    "unit_id",
    "window"
  ]
}
```

### First-run output and recovery record

```json
{
  "errors": {
    "bare_constructor": {
      "type": "ValueError",
      "message": "[E1006] Environment cannot be constructed directly \u2014 use a factory method.\n\nMost common (from positions you recorded):\n    env = Environment.from_samples(positions, bin_size=2.0)\n\nOther factories, chosen by the data you have:\n    from_polygon     \u2014 a Shapely polygon boundary\n    from_graph       \u2014 a track/maze graph (linearized 1D)\n    from_grid_mask   \u2014 an N-D boolean mask + grid edges\n    from_pixel_mask  \u2014 a 2D image / pixel mask\n\nAvoid:\n    env = Environment()  # not supported\n\nSee each factory's docstring for its exact arguments, or:\n    https://edeno.github.io/neurospatial/errors/#e1006-environment-constructed-directly\nWhy: a factory builds the geometry and connectivity required by Environment.\nFix: env = Environment.from_samples(positions, bin_size=2.0)"
    },
    "no_bins": {
      "type": "ValueError",
      "message": "[E1001] No active bins found after filtering.\n\nDiagnostics:\n  Data range: [(7.723547903720695, 62.478921530960555), (2.3915818228210464, 62.48281260948955)]\n  Data extent: [54.75537362723986, 60.091230786668504]\n  Number of samples: 6000\n  bin_size: 5.0\n  Grid shape: (12, 14)\n  Total bins in grid: 168\n  bin_count_threshold: 1000000\n  Morphological operations: dilate=False, fill_holes=False, close_gaps=False\n\nCommon causes:\n  1. bin_size is too large relative to your data range\n  2. bin_count_threshold is too high (no bins have enough samples)\n  3. Data is too sparse and morphological operations are disabled\n\nSuggestions to fix:\n  1. Reduce bin_size to create more bins\n  2. Reduce bin_count_threshold (try 0 for initial testing)\n  3. Enable morphological operations (dilate=True, fill_holes=True, close_gaps=True)\n  4. Check that positions cover the expected spatial range"
    },
    "not_linearized": {
      "type": "AttributeError",
      "message": "to_linear() is only available for 1D environments (GraphLayout). This environment is 2D. Use bin_at() to map positions to bins for N-D environments."
    },
    "missing_frame_times": {
      "type": "TypeError",
      "message": "EnvironmentVisualization.animate_fields() missing 1 required keyword-only argument: 'frame_times'"
    },
    "bad_positions": {
      "type": "ValueError",
      "message": "compute_spatial_rate: positions has shape (6000, 1) but env is 2-D, so positions must have shape (n_samples, 2)\nWhy: timestamps in seconds and sample-aligned coordinates are needed to assign observations to the correct bins.\nFix: pass all coordinates, e.g. np.column_stack([x, y]); for a 1-D track, build env from 1-D data (positions[:, None]) or Environment.linear_track(...)"
    },
    "batch_plot_no_unit": {
      "type": "ValueError",
      "message": "plot() requires a unit index: this batch result holds 6 units, so there is no single rate map to plot. Pass a unit index, e.g. result.plot(0), or iterate the units with `for r in result: r.plot()` / `result[i].plot()`."
    },
    "single_xarray": {
      "type": "AttributeError",
      "message": "'SpatialRateResult' object has no attribute 'to_xarray'"
    },
    "tiny_grid": {
      "type": "UserWarning",
      "message": "Creating large grid with shape (1113, 1113) (1,238,769 bins). Estimated memory usage: 1318.4 MB.\nWhy: allocating this grid may consume substantial memory.\nFix: increase bin_size; this grid has 1,238,769 bins. For example, use bin_size=2.0 in the same units as positions, or infer_active_bins=True"
    }
  },
  "warnings": [
    {
      "type": "UserWarning",
      "message": "bin_size=1000.0 is at least the per-axis data extent [54.75537362723986, 60.091230786668504] in every dimension.\nWhy: this produces very few bins and may indicate a units mismatch.\nFix: use bin_size=1.20182 (in the same units as positions), or choose a size below the data extent"
    },
    {
      "type": "UserWarning",
      "message": "47/47 spike_times (100%) fell outside the position time window [0, 59.99]; spike_times.min()=1420 spike_times.max()=59760. Check that spike_times and times share units (both seconds). Dropped spikes do not contribute. Set warn_on_drop=False to suppress this warning."
    }
  ],
  "first_field_summary": {
    "n_bins": 78,
    "peak_firing_rate": 5.321146936118858,
    "total_occupancy": 59.99,
    "spike_window_assumed": true,
    "spike_window": null,
    "method": "diffusion_kde"
  },
  "single_repr": "SpatialRateResult(n_bins=78, peak_firing_rate=5.321, total_occupancy=59.99, spike_window_assumed=True, spike_window=None, method=diffusion_kde)",
  "population_repr": "SpatialRatesResult(n_bins=78, peak_firing_rate=19.02, total_occupancy=59.99, spike_window_assumed=True, spike_window=None, n_neurons=6, method=diffusion_kde)",
  "population_summary": {
    "n_bins": 78,
    "peak_firing_rate": 19.018433354844824,
    "total_occupancy": 59.99,
    "spike_window_assumed": true,
    "spike_window": null,
    "n_neurons": 6,
    "method": "diffusion_kde"
  },
  "single_columns": [
    "peak_x",
    "peak_y",
    "peak_rate",
    "method"
  ],
  "population_columns": [
    "peak_x",
    "peak_y",
    "peak_rate",
    "spatial_info",
    "sparsity",
    "grid_score",
    "border_score",
    "cell_type",
    "method"
  ],
  "population_index": [
    100,
    101,
    102,
    103,
    104,
    105
  ],
  "table_attrs": {},
  "dense_shape": [
    468,
    7
  ],
  "peak_location": [
    27.631813013935638,
    25.46313677472652
  ],
  "spatial_information": 1.1763165129014044,
  "figure_axes": 2,
  "colormap": "viridis",
  "source_reads": []
}
```

### Public-documentation discovery log

README; quickstart; public API index; interoperability, workflow, trajectory/behavioral and animation guides; animation overlays; public examples 05, 15, 20, 21, 24, 26, 27. Public runtime docs: Environment factories/queries/plots, readers/intervals, encoding result/mapping/predicates, segmentation labels, decoder/count helpers, assembly/EV functions/results, event helpers, SimulationSession/convenience helpers and egocentric transforms. No implementation or test file was opened.

The exact independent attempts and raw logs were retained in `/private/tmp/neurospatial-checkpoint-nwb-track` and `/private/tmp/neurospatial-checkpoint-ego-stats-events`; lead scratch files are in `/private/tmp/neurospatial-checkpoint`. The scripts and key durable outputs above allow reproduction without those scratch directories.

### Exact first-run and recovery probe

This uses the synthetic NWB fixture above and records the output JSON; no implementation source is read.

```python
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO
from neurospatial import Environment
from neurospatial.io.nwb import read_position, read_units
from neurospatial.encoding import compute_spatial_rate, compute_spatial_rates

with NWBHDF5IO("session.nwb", "r") as io:
    nwb = io.read()
    positions, times = read_position(nwb)
    spikes, ids = read_units(nwb)

errors = {}
for name, operation in (
    ("bare_constructor", lambda: Environment()),
    ("no_bins", lambda: Environment.from_samples(positions, bin_size=5.0, bin_count_threshold=10**6)),
):
    try:
        operation()
    except Exception as error:
        errors[name] = {"type": type(error).__name__, "message": str(error)}

env = Environment.from_samples(positions, bin_size=5.0, units="cm")
single = compute_spatial_rate(env, spikes[0], times, positions)
population = compute_spatial_rates(env, spikes, times, positions, unit_ids=ids)
for name, operation in (
    ("not_linearized", lambda: env.to_linear(positions[:1])),
    ("missing_frame_times", lambda: env.animate_fields(np.ones((2, env.n_bins)), backend="html")),
    ("bad_positions", lambda: compute_spatial_rate(env, spikes[0], times, positions[:, :1])),
    ("batch_plot_no_unit", lambda: population.plot()),
    ("single_xarray", lambda: single.to_xarray()),
):
    try:
        operation()
    except Exception as error:
        errors[name] = {"type": type(error).__name__, "message": str(error)}

with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter("always")
    coarse = Environment.from_samples(positions, bin_size=1000.0, units="cm")
    wrong_time = compute_spatial_rate(env, spikes[0] * 1000.0, times, positions)
warning_messages = [{"type": w.category.__name__, "message": str(w.message)} for w in seen]

# Turn the observed huge-grid UserWarning into an exception before allocation.
with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    try:
        Environment.from_samples(np.array([[0.0, 0.0], [100.0, 100.0]]), bin_size=0.09)
    except Exception as error:
        errors["tiny_grid"] = {"type": type(error).__name__, "message": str(error)}

ax = single.plot()
single_table = single.summary_table()
table = population.summary_table()
output = {
    "errors": errors, "warnings": warning_messages,
    "first_field_summary": single.summary(), "single_repr": repr(single),
    "population_repr": repr(population), "population_summary": population.summary(),
    "single_columns": single_table.columns.tolist(), "population_columns": table.columns.tolist(),
    "population_index": table.index.tolist(), "table_attrs": table.attrs,
    "dense_shape": list(population.to_dataframe().shape),
    "peak_location": single.peak_location().tolist(),
    "spatial_information": float(single.spatial_information()),
    "figure_axes": len(ax.figure.axes), "colormap": ax.collections[0].get_cmap().name,
    "source_reads": [],
}
Path("ux-probes.json").write_text(json.dumps(output, indent=2, default=str))
Path("ux-table.txt").write_text(str(table) + "\n" + table.to_string())
plt.close("all")
print("Completed first-run, recovery, and output probes.")
```
