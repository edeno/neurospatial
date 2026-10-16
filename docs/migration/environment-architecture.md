# Environment architecture changes

The `Environment` internals were reorganized so that each kind of data has one
owner. Analyses you run with `Environment` do not change. This page lists what
you may need to change in your own code.

## Saving environments: use files, not pickles from older versions

Pickle files of an `Environment` written by an earlier version of neurospatial
cannot be loaded. Pickle still works for sending an environment to worker
processes of the same version (for example with `multiprocessing`), but it is
not meant for storing data.

To keep an environment between sessions, save it with `to_file` (or write it to
an NWB file) and load it back with `from_file`:

```python
env.to_file("arena")
env = Environment.from_file("arena")
```

If you only have an old pickle, load it with the neurospatial version that
wrote it, call `env.to_file(...)` there, and then load that file with the
current version.

## Removed import paths

These private modules no longer exist. Import from the real location, or use the
public API.

| Removed | Use instead |
| --- | --- |
| `neurospatial.environment._coordinates` | `neurospatial.geometry._coordinates` (or `Environment.coordinate_system`) |
| `neurospatial.environment._metadata` | `Environment.name`, `.units`, `.frame` |
| `neurospatial.environment._ownership` | `neurospatial.geometry._ownership` |
| `neurospatial.environment._from_bins` | `Environment.from_bins` |
| `neurospatial.environment.transforms` | `Environment.subset`, `Environment.rebin` |
| `neurospatial.layout._binning`, `_locators`, `_polygon`, ... | `neurospatial.geometry` (same module names) |
| `neurospatial.layout._capabilities` (`RectilinearGridProvider`) | `layout.binning().rectilinear` |

`neurospatial.arena` still provides the public arena classes (`Arena`,
`OpenArena`, `MaskArena`, `TrackArena`, `UnknownArena`, `EgocentricSurround`).

## Shapely is required

`neurospatial.environment._HAS_SHAPELY` is removed. Shapely is a required
dependency and is always available.

## Environments are immutable values

An `Environment` cannot be changed after it is built, so it now behaves like a
plain value.

**Equality.** Two environments are equal only when they are the same object.
`env == "session1"` (comparing with the name) no longer works; compare
`env.name == "session1"`. `env.with_metadata(name="other")` returns a new
environment, which does not equal the original. Environments can be put in sets
and used as dictionary keys.

**Copying.** `Environment.copy()` is removed because there is nothing to copy.
`copy.copy(env)` and `copy.deepcopy(env)` return the same object.

| Before | Now |
| --- | --- |
| `env2 = env.copy()` | `env2 = env` (or `env.with_metadata()` for a separate object with its own empty caches) |
| `env.copy()` then `with_regions(...)` | `env.with_regions(...)` |

**No "fitted" state.** An environment always has its bins and connectivity, so
`EnvironmentNotFittedError` (error code E1004), `check_fitted` and the
`neurospatial.environment.decorators` module are removed. Calling
`Environment()` directly still raises the E1006 error that points to the
factories. Passing an object that is not an `Environment` to an analysis
function raises a `TypeError` that shows the corrected call.

**Linearization errors.** `to_linear` and `linear_to_nd` raise `TypeError`
(not `AttributeError`) on an environment that is not a linearized track.

## Polar environments report their own layout type

`Environment.from_polar_egocentric(...)` environments now report
`env.layout_type == "Polar"` (it was `"MaskedGrid"`), and `print(env)` shows
`Polar`. The bins are sectors in distance and egocentric angle, so the old
name suggested a Cartesian grid. To test whether an environment is polar, use
the coordinate kind rather than the layout name:

```python
if env.coordinate_kind == "polar":
    ...
```

A polar environment saved with an earlier version loads unchanged and keeps
the layout name stored in its file.

Whether the angle wraps around at ±π is read only from the environment's
`EgocentricSurround` arena. Files from before arenas were saved
(`Environment-v2` files and NWB environment schema 2.0) do not have one, so a
polar environment stored in them can no longer be read. Load it with the
version that wrote it and save it again, or rebuild it with
`Environment.from_polar_egocentric(..., circular_angle=...)`.

## Custom layout engines supply `binning()`

Each layout engine now hands its geometry to the environment through one
method, `binning()`, which returns a `Binning`: the bins in connectivity edge
order, point lookup, finite-volume faces (or the reason they are missing) and
any grid, track, shape and outline records. All built-in engines implement it,
and the `LayoutEngine` protocol lists it.

If you wrote your own layout engine:

- Implement `binning()` from the engine's built state. Returning only built-in
  records is required, because the environment keeps a copy of the record and
  never calls back into the engine. A subclass of a built-in engine inherits a
  working `binning()`; such subclasses used to be refused and now publish
  normally.
- The environment keeps only what `binning()` returns (plus `connectivity`,
  `dimension_ranges` and the grid attributes). It never calls
  `point_to_bin_index` or `finite_volume_geometry`, so to change lookup or
  faces, change what `binning()` returns. A layout whose methods were replaced
  on the instance is no longer refused.
- An external object that has no `binning()` method still needs an installed
  snapshot adapter (see the interoperability guide).
- `neurospatial.layout._capabilities.RectilinearGridProvider` is removed. Read
  the grid record from `layout.binning().rectilinear`.
