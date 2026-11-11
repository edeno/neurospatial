# Carey, Tanaka & van der Meer (2019): the authors' sequence pipeline in Octave

This folder runs the code that made the paper's decoded-sequence counts on the
DataLad MotivationalT data, so neurospatial's route can be checked against the
authors' own output. The output is committed, so the tests don't need Octave:
`tests/data/carey_2019/authors_octave_sequences.csv` and `authors_octave_sessions.json`.

## What runs

- `Carey_etal_submitted/decoding_Seq/Generate_DecSeqShuf.m` and
  `getEligibleSequences.m` from github.com/vandermeerlab/papers at `3aebc8e`,
  with the cfg of `ALL_Generate_DecSeq.m` and shuffles off (`n_shuffles=[]`).
- Their helpers from github.com/vandermeerlab/vandermeerlab at `ad0bbd4`, the
  commit the paper's README pins (`code-matlab/shared`, `tasks/Alyssa_Tmaze`).
- The paper's own candidate events (`SWRcandidates/<session>-candidates.zip`).

## Stand-ins for Octave (`shims/`)

| File | Replaces | Note |
| --- | --- | --- |
| `Nlx2MatVT.m`, `Nlx2MatEV.m` | Neuralynx MEX readers | read the `.mat` written by `export_nlx.py` from the raw records |
| `nearest_idx3.m` | `nearest_idx3.c` (MEX) | port of the C; checked on the edge cases |
| `nanmean.m`, `nansum.m`, `nanstd.m` | statistics toolbox | MATLAB semantics: an all-NaN sum is 0, mean NaN |
| `regress.m` | statistics toolbox | slope and `[R^2 F p errvar]` only; NaN for a constant series |
| `confusionmat.m` | statistics toolbox | used only by `DecodeErrorZ.m` (a diagnostic) |

## Run

```bash
git clone https://github.com/vandermeerlab/papers && git -C papers checkout 3aebc8e
git clone https://github.com/vandermeerlab/vandermeerlab && git -C vandermeerlab checkout ad0bbd4
export CAREY_VDM_CODE=$PWD
# inside the neurospatial environment (export_nlx.py imports it):
./run_all.sh /tmp/carey_octave_work          # about 5 min and 3-9 GB per session, 5 at a time
python export_reference.py /tmp/carey_octave_work ../../../tests/data/carey_2019
```

`run_all.sh` reads `keys.m` as text first and refuses a session whose file holds
anything but `ExpKeys` assignments, because the authors' code runs it.

## Results (Octave 11.3, arm64, 2026-11-10)

| Rat | Octave | Paper (Supp. Fig. 6) |
| --- | --- | --- |
| R042 | 1,210 | 1,257 |
| R044 | 91 | 95 |
| R050 | 393 | 429 |
| R064 | 1,963 | 1,999 |
| all | 3,657 | 3,780 |

- **The identity-shuffle filter removes nothing.** `Generate_DecSeqShuf.m`'s
  `shuf_Q(shuf_mask)` uses row indices as linear indices, so every shuffled bin
  is a permutation of the session's first 5 ms bin; with 3 shuffles on
  R050-2014-03-29 no sequence was removed (one detection would have removed one).
- **The time grid.** Octave on arm64 builds `firstSpike:dt:lastSpike+dt` element
  by element as `fma(k, dt, first)` (0 mismatches over ~1 M rows); MATLAB builds
  it from both ends with plain arithmetic (MathWorks' `colonop.m`). The 50 ms
  length test sees the last bit: with MATLAB's grid the total is 3,660.
- **Unexplained:** the remaining 3% (3,660 against 3,780). Each rat's paper
  count lies between our strict count and the count that accepts every
  exactly-50 ms sequence (3,887), but no step of the code was found to explain
  it. A MATLAB run would settle whether it is the environment or the data.
- **An artifact of the code (kept in the reference, not in neurospatial):** a
  position bin with no occupancy has a NaN tuning curve; `DecodeZ.m`'s `nansum`
  gives it a likelihood of one and `max` takes the first such bin on a tie, so
  rows whose occupied bins all score below one decode to bin 1 and stay there.
  264 of the 3,657 sequences sit at bin 1 throughout.

`tests/datasets/carey_authors_route.py` ports the same pipeline to Python; the
realdata tests check it against this reference.
