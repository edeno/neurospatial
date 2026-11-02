# Designs — silent-result guards

[← back to PLAN.md](PLAN.md)

Each section gives the detection rule, the reason for its threshold, and the
code. The message format follows
[shared-contracts.md#discarded-data-warning](shared-contracts.md#discarded-data-warning).

- [S1 positions outside the environment](#s1-positions-outside-the-environment)
- [S2 boolean-masked samples](#s2-boolean-masked-samples)
- [S3 angular widths in degrees](#s3-angular-widths-in-degrees)
- [S4 zero-filled rate arrays](#s4-zero-filled-rate-arrays)
- [S5 decode dt vs times](#s5-decode-dt-vs-times)
- [S6 spikes outside the recording](#s6-spikes-outside-the-recording)
- [S7 swapped metric arguments](#s7-swapped-metric-arguments)
- [S8 laps on a track without a loop](#s8-laps-on-a-track-without-a-loop)
- [S9 spatial bandwidth below the bin size](#s9-spatial-bandwidth-below-the-bin-size)

## S1 positions outside the environment

**Today.** `IntervalGates.narrowed(keep)` (`_intervals.py:1192`) ANDs the
bin-gate mask into `mask` and loses which intervals the bin gate alone
removed. `warn_excluded_intervals` (`_intervals.py:1699`) returns early when
any interval survives (`:1745`).

**Change.**

- Record the intervals that other gates kept but the bin gate dropped. Add a
  field to `IntervalGates`:

  ```python
  bin_dropped: NDArray[np.bool_] | None = None  # (n_samples - 1,): kept by other gates, dropped by the bin check
  ```

  `narrowed` sets it:

  ```python
  def narrowed(self, keep):
      dropped = self.mask & ~keep
      if self.bin_dropped is not None:
          dropped |= self.bin_dropped
      return replace(
          self,
          mask=self.mask & keep,
          start_bin_gate=self.start_bin_gate or bool(self.mask.any()),
          bin_dropped=dropped,
      )
  ```

- In `warn_excluded_intervals`, before the `if gates.mask.any():` branch,
  compute the discarded duration:

  ```python
  if gates.mask.any() and gates.bin_dropped is not None:
      dt = np.diff(gates.times)
      lost = float(dt[gates.bin_dropped].sum())
      kept_by_others = float(dt[gates.mask | gates.bin_dropped].sum())
      if kept_by_others > 0 and lost / kept_by_others > DISCARD_WARN_FRACTION:
          warnings.warn(_outside_env_message(gates, lost, kept_by_others, samples), UserWarning,
                        stacklevel=_external_stacklevel())
          return
  ```

- `_outside_env_message` needs the environment extent and the position
  extent. Pass them through a new optional `extent: tuple[str, str] | None`
  keyword, preformatted by the caller as `"x∈[-1, 101] cm"`. Callers without
  positions pass `None`, and the message then omits the range.
- Callers: the 11 `warn_excluded_intervals(` sites. Only callers that apply a
  bin gate (`narrowed`) need to pass `extent`.

**Threshold.** 5% of the time the other gates kept. A real recording with
brief tracking dropouts loses well under 5%. The 8×-pixel case in the review
loses 94%.

## S2 boolean-masked samples

**Signature of a masked array.** Boolean masking removes whole samples, so
the remaining steps are whole multiples (≥ 2) of the median sampling
interval. Dropped camera frames show the same pattern, but they are rare and
spread out, whereas a speed mask removes long runs.

**Rule.** Let `step = median(diff(times))` and `r = diff(times) / step`. Count
an interval as "skipped" when `r ≥ 1.5` and `|r − round(r)| < 0.1`, and it is
**not** already a gap (`diff(times) ≤ max_gap`). The skipped time is
`(r − 1) · step` per interval. Warn when the total skipped time exceeds
`DISCARD_WARN_FRACTION` of the session duration:

```python
def _masked_sample_fraction(times, max_gap):
    dt = np.diff(times)
    if dt.size < 10:
        return 0.0
    step = float(np.median(dt))
    if not np.isfinite(step) or step <= 0:
        return 0.0
    r = dt / step
    skipped = (r >= 1.5) & (np.abs(r - np.round(r)) < 0.1)
    if max_gap is not None:
        skipped &= dt <= max_gap
    return float(((r[skipped] - 1.0) * step).sum() / (times[-1] - times[0]))
```

**Where.** In `warn_excluded_intervals`, on the kept-data branch, after S1.

**Message.**

- **What:** "about {pct}% of the recording is missing in whole-sample steps
  shorter than max_gap, which looks like times/positions were boolean-masked."
- **Why:** "those steps are bridged as continuous occupancy, so masked-out
  time and spikes are still counted."
- **Fix:** "pass the full arrays and use min_speed= or epochs=; or set
  max_gap below the sampling interval."

**False-positive check.** Test on 30 Hz times with 0.5% of frames dropped at
random. It must not warn.

## S3 angular widths in degrees

**Where.** Rename `_check_smoothing_bandwidth` (`encoding/directional.py:2230`)
to `_check_angular_widths(bin_size, bandwidth, angle_unit)` and keep its
single call site (`:2322`).

**Rules**, applied after converting to radians:

- A `bin_size` above π refuses: it leaves fewer than two bins. Fix: "pass
  angle_unit='deg' if {bin_size} is in degrees."
- A `bandwidth` above 1 rad (57°) warns: "a circular Gaussian this wide
  flattens any head-direction tuning." Fix: same as `bin_size`.

**Threshold reason.** A published HD tuning width is about 60–90° full width
(≈0.5–0.8 rad SD). A smoothing SD of 1 rad or more is wider than the tuning it
smooths.

## S4 zero-filled rate arrays

**Where.** `_prepare_decode_inputs` (`decoding/posterior.py:553`), on the
bare-array path (no `spatial_support`, no `prior`).

**Rule.** A bin where **every** neuron's rate is exactly 0 has no training
evidence, unless the user made it so on purpose. Warn when such bins make up
more than 5% of `env.n_bins`:

```python
all_zero = ~np.any(models != 0.0, axis=0)
if spatial_support is None and prior is None and all_zero.mean() > DISCARD_WARN_FRACTION:
    warnings.warn(_format_error(
        f"{context}: {int(all_zero.sum())} of {env.n_bins} bins have rate 0 for every neuron.",
        why="zero-filled bins usually mark missing rate estimates; the default "
            "uniform prior gives them full weight, and a silent population "
            "favors them.",
        fix="pass the rate result itself (decode_position(env, counts, rates, dt=dt)), "
            "or prior=rates.spatial_support * env.bin_sizes.",
    ), UserWarning, stacklevel=_external_stacklevel())
```

## S5 decode dt vs times

**Where.** In `decode_position` and `decode_position_summary`, when `times`
is given, before `_prepare_decode_inputs`. Also in `bin_spikes_in_time`
(`decoding/_time_bins.py:104`).

**Rules.**

- `decode_position(times=…)`: refuse when `dt` and
  `np.median(np.diff(times))` differ by more than 1%. The bin width is part
  of the likelihood, so a mismatch is never correct.
- `bin_spikes_in_time`: warn when `dt > 1.0` s and every spike time is below
  `1000 · dt`. Use the same "probably milliseconds" wording as the peri-event
  window heuristic.

## S6 spikes outside the recording

Replace `_DROP_WARN_THRESHOLD` (`encoding/_spatial_counts.py:68`) with
`DISCARD_WARN_FRACTION`. Both `_emit_*_warning` functions (`:1072`, `:1116`)
keep their messages.

Remove the `_DROP_WARN_THRESHOLD` comment at `:66–67`. The 100%-dropped case
is still covered, because 1.0 > 0.05.

## S7 swapped metric arguments

Make `occupancy` keyword-only in:

- `spatial_information` (`encoding/_metrics.py:168`)
- `sparsity` (`:736`)
- `selectivity` (`:1198`)
- `information_per_second` (`:1301`)

There is no runtime heuristic: rates and occupancy share a shape and both
are non-negative.

## S8 laps on a track without a loop

**Where.** `detect_laps` (`behavior/segmentation.py:888`).

**Rule.** When `method` resolves to a loop method and the environment's
connectivity graph has no cycle (`nx.cycle_basis(env.connectivity)` is
empty), raise `ValueError`:

- **What:** "detect_laps counts laps around a loop, but this environment has
  no loop."
- **Fix:** "for back-and-forth runs on a linear track or arm, use
  detect_trials(env, times, positions, regions=[start, end])."

Check first whether `method="reference"` with `reference_lap=` is meaningful
on a linear track. If it is, apply the rule only to the loop methods.

## S9 spatial bandwidth below the bin size

**Where.** Next to the existing bandwidth validation in
`encoding/_smoothing.py:751-766`. That validation refuses `bandwidth <= 0`,
except that `method="binned"` allows 0.

**Rule.** Warn when an explicit positive `bandwidth` is below a quarter of
the bin length:

- **What:** "bandwidth={b} is far below the {length} bin size, so almost no
  smoothing happens."
- **Why:** "a bandwidth in different units (for example metres on a cm
  environment) looks like this."
- **Fix:** "pass bandwidth in env.units, or method='binned' with
  bandwidth=0 for no smoothing."

**Bin length.**

- `env.bin_sizes` (`environment/core.py:4736`) is each bin's area or volume,
  not a length. Use `np.median(env.bin_sizes) ** (1 / env.n_dims)`, unless
  the environment already exposes a length accessor (check for one first;
  `length_scale` is an open follow-up).
- `method="binned"` with `bandwidth=0` must stay silent.
