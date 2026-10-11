#!/usr/bin/env bash
# Mutation harness for the metrics work (2026-10-11).
#
# Lessons this script is built around, both learned the hard way in earlier
# sessions (see EXPANSION_QUEUE.md, Grad-CAM and canonical-losses entries):
#   1. Take the baseline ONCE, BEFORE any mutation, and ASSERT it is pristine
#      by grepping for a line that exists only in the correct implementation.
#      A snapshot taken after a mutation is already broken, and every restore
#      then writes the mutant back over the good code.
#   2. Verify every `sed` actually applied with `diff -q` before building. A
#      no-op sed is indistinguishable from a surviving mutant.
#   3. Report compile failure as MUT-BROKEN, never MUT-CAUGHT. A mutant that
#      does not compile proves nothing about test strength.
#   4. Never run `make` against this tree while the harness holds a mutant --
#      the build would cache the mutant in a .o. The tree is restored and
#      rebuilt between every mutation instead.

set -u
cd "$(dirname "$0")/.." || exit 1
SRC=include/nn/utils/metrics.cpp
BASE=$(mktemp -u /tmp/metrics_baseline_XXXXXX.cpp)
TMP=$(mktemp -u /tmp/metrics_mutant_XXXXXX.cpp)
trap 'cp -f "$BASE" "$SRC"; rm -f "$BASE" "$TMP"' EXIT

# --- 1. baseline, taken once, and asserted pristine -------------------------
cp "$SRC" "$BASE"
# These lines exist ONLY in the correct implementation. If any is missing the
# snapshot is not pristine and the whole run is void.
while IFS= read -r needle; do
  if ! grep -qF "$needle" "$BASE"; then
    echo "ABORT: baseline is not pristine -- missing: $needle"
    exit 2
  fi
done <<'NEEDLES'
const double d2_fps = c.fps[i - 1] - 2.0 * c.fps[i] + c.fps[i + 1];
halve = (y_proba.cols < 3);
sum += d0 * d0 + d1 * d1;
sum += (c.recall[i + 1] - c.recall[i]) * c.precision[i];
if (interior[mid] < probs[i]) lo = mid + 1;
if (bin_count[k] == 0) continue;
NEEDLES
echo "baseline verified pristine"

pass=0; fail=0
run_mutation() {
  local name="$1"; shift
  cp -f "$BASE" "$SRC"
  # --- 2. apply the mutation, verifying it landed --------------------------
  sed "$@" "$SRC" > "$TMP"
  if diff -q "$SRC" "$TMP" >/dev/null 2>&1; then
    echo "MUT-NOOP   $name  (sed changed nothing -- not a real mutation)"
    fail=$((fail+1))
    return
  fi
  cp -f "$TMP" "$SRC"

  # --- 3. build. Compile failure is NOT a catch. ---------------------------
  if ! make -s build/test_metrics >/tmp/mut_build.log 2>&1; then
    echo "MUT-BROKEN $name  (compile failure -- proves nothing)"
    fail=$((fail+1))
    cp -f "$BASE" "$SRC"
    return
  fi
  if ./build/test_metrics >/tmp/mut_run.log 2>&1; then
    echo "MUT-SURVIVED $name  <-- GAP"
    fail=$((fail+1))
  else
    local n sig
    n=$(grep -c '^\[FAIL\]' /tmp/mut_run.log)
    # Distinguish a clean assertion failure from an abort/segfault. An
    # uncaught std::invalid_argument or a core dump IS a real catch -- the
    # mutant broke the function contract -- but it is a different KIND of
    # catch and the distinction has to stay visible, because "MUT-CAUGHT (by 0
    # assertions)" on a segfault is indistinguishable from a harness artifact.
    sig=""
    if [ "$n" -eq 0 ]; then
      if grep -q "terminate called\|Aborted\|core dumped" /tmp/mut_run.log; then
        sig=" [abort: threw past the test harness, 0 assertion failures]"
      else
        sig=" [signal: crashed, 0 assertion failures]"
      fi
    fi
    echo "MUT-CAUGHT $name  (by $n assertions)$sig"
    pass=$((pass+1))
  fi
  cp -f "$BASE" "$SRC"
}

# M1: drop the +inf/(0,0) origin prepend in roc_curve.
run_mutation "M1 drop ROC origin prepend" \
  's|    out.fpr.push_back(0.0);|    /* mutated */|; s|    out.tpr.push_back(0.0);|    /* mutated */|'

# M2: emit one sweep point per SAMPLE instead of per distinct score.
# The `|| true` form is deliberate: replacing the whole condition with `false`
# also removes the short-circuit that guards sorted_scores[i+1], so that mutant
# SEGFAULTS on the out-of-bounds read instead of failing an assertion. A crash
# is a much weaker signal than a specific failed assertion -- it cannot tell a
# real catch from a harness artifact -- and the earlier `false` form scored
# "MUT-CAUGHT (by 0 assertions)" on the strength of a core dump alone.
# `last || true` keeps the bounds guard intact while still emitting every
# sample, so the mutant fails on an assertion instead.
run_mutation "M2 per-sample instead of per-distinct-score" \
  's|        if (last \|\| sorted_scores\[i\] != sorted_scores\[i + 1\]) {|        if (last \|\| true) {|'

# M3: replace the AP STEP integral with the trapezoid area under the PR curve.
run_mutation "M3 AP step integral -> trapezoid" \
  's|        sum += (c.recall\[i + 1\] - c.recall\[i\]) \* c.precision\[i\];|        sum -= 0.5 * (c.recall[i] - c.recall[i + 1]) * (c.precision[i] + c.precision[i + 1]);|'

# M4: flip the Brier halving rule to cols < 2 (was cols < 3).
run_mutation "M4 brier halving rule cols<3 -> cols<2" \
  's|        default: halve = (y_proba.cols < 3); break;|        default: halve = (y_proba.cols < 2); break;|'

# M5: remove the log-loss clip entirely.
run_mutation "M5 remove log-loss epsilon clip" \
  's|            p = std::min(std::max(p, eps), 1.0 - eps);|            p = p;|g'

# M6: flip the calibration bin-edge assignment from left- to right-closed.
run_mutation "M6 searchsorted side left -> right" \
  's|            if (interior\[mid\] < probs\[i\]) lo = mid + 1;|            if (interior[mid] <= probs[i]) lo = mid + 1;|'

# M7: stop dropping empty calibration bins.
run_mutation "M7 keep empty calibration bins" \
  's|        if (bin_count\[k\] == 0) continue;|        /* mutated: keep empty bins */|'

# M8: make weighted OVR averaging divide by num_classes instead of support.
run_mutation "M8 weighted averaging -> divide by num_classes" \
  's|    return weighted / support_total;|    return weighted / static_cast<double>(per_class.size());|'

# M9: drop the collinear-point filter (changes curve shape, never the area).
run_mutation "M9 disable drop_collinear filter" \
  's|^    drop_collinear(c);|    // mutated: filter removed|'

# M10: the binary Brier form scores only the target column instead of both.
run_mutation "M10 binary Brier: one column instead of two" \
  's|            sum += d0 \* d0 + d1 \* d1;|            sum += d1 * d1;|'

echo
echo "=== $pass caught, $fail not caught ==="
cp -f "$BASE" "$SRC"
[ "$fail" -eq 0 ]