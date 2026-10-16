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

`neurospatial.arena` still provides the public arena classes (`Arena`,
`OpenArena`, `MaskArena`, `TrackArena`, `UnknownArena`, `EgocentricSurround`).

## Shapely is required

`neurospatial.environment._HAS_SHAPELY` is removed. Shapely is a required
dependency and is always available.
