#!/bin/bash
# Run the authors' decoded-sequence pipeline (Carey, Tanaka & van der Meer 2019)
# in Octave on the paper's 19 sessions, shuffles off. See README.md.
#
# usage: CAREY_VDM_CODE=<clones> ./run_all.sh <work-dir>
# Needs: octave-cli on PATH, the neurospatial carey_2019 cache (load each session
# once with neurospatial.datasets.load) and the paper's candidate files
# (tests/datasets/test_carey_2019.py downloads them to paper_candidates/).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mkdir -p "${1:?work dir}" && cd "$1" && pwd)"
CACHE="${NEUROSPATIAL_DATA_DIR:-$HOME/.cache/neurospatial/datasets}/carey_2019"
: "${CAREY_VDM_CODE:?set CAREY_VDM_CODE to the folder holding papers/ and vandermeerlab/}"
mkdir -p "$WORK/sessions" "$WORK/logs"
shopt -s nullglob
for cand in "$CACHE"/paper_candidates/*-candidates.mat; do
  s=$(basename "$cand" -candidates.mat)
  d="$WORK/sessions/$s"; mkdir -p "$d/files"
  keys=("$CACHE/$s"/*keys.m)
  # keys.m is MATLAB that the code runs: refuse anything but comments and ExpKeys assignments.
  if grep -v -E "^\s*(%.*)?$|^\s*ExpKeys\.[A-Za-z_0-9]+(\{[0-9]+\})?\s*=[^;]*;\s*(%.*)?$" "${keys[0]}" | grep -q .; then
    echo "$s: keys.m holds more than assignments; not running it" >&2; continue
  fi
  for f in "$CACHE/$s"/*.t "$CACHE/$s"/*._t "$CACHE/$s"/*keys.m "$CACHE/$s"/*metadata.mat \
           "$CACHE/$s"/*Events.nev "$CACHE/$s"/*.nvt "$CACHE/$s"/*times.mat; do
    ln -sf "$f" "$d/"
  done
  ln -sf "$cand" "$d/"
  [ -f "$d/VT_export.mat" ] || python -I "$HERE/export_nlx.py" "$d" "$d"
done
one() {
  d=$1; s=$(basename "$d")
  [ -f "$d/files/octave_result.mat" ] && { echo "$s done already"; return; }
  (cd "$HERE" && octave-cli --no-gui --eval "session_dir='$d'; n_shuffles=[]; run_session") > "$WORK/logs/$s.log" 2>&1 \
    && echo "$s $(grep -m1 'eligible left' "$WORK/logs/$s.log")" || echo "$s FAILED (see logs/$s.log)"
}
n=0
for d in "$WORK"/sessions/*/; do
  one "${d%/}" &
  n=$((n + 1)); if [ $((n % 5)) -eq 0 ]; then wait; fi
done
wait
