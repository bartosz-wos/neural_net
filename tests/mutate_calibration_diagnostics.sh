#!/usr/bin/env bash
# mutate_calibration_diagnostics.sh — mutation harness for the calibration
# diagnostics in include/nn/utils/metrics.cpp.
#
# Lessons applied from the Grad-CAM and metrics sessions:
#   * Take the baseline ONCE, before any mutation, and ASSERT it is pristine
#     by grepping for a line that exists only in the correct implementation.
#     A harness that snapshots while a mutation is already applied will
#     eventually restore a mutant over correct code.
#   * Verify each sed ACTUALLY APPLIED (diff -q) before building. A no-op sed
#     is indistinguishable from a mutation that survived.
#   * Report compile failure as MUT-BROKEN, never as MUT-CAUGHT. A mutant that
#     fails to build proves nothing about test strength.
#   * Line-anchored mutations only. `sed '0,/X/s//Y/'` mutates the FIRST match
#     in the file, which in metrics.cpp is usually the wrong function.
set -u

REPO=/home/stefan/neural_net
SRC="$REPO/include/nn/utils/metrics.cpp"
HDR="$REPO/include/nn/utils/metrics.h"
# RELATIVE target, and driven with the repo as the working directory.
# An ABSOLUTE target here silently does nothing: `make /abs/path/build/test_x`
# reports "up to date" and skips the rebuild, so every mutation would run the
# STALE binary and report MUT-SURVIVED. That is how the first run of this
# harness reported 7/7 survived when not one mutation had been compiled in.
TARGET=build/test_calibration_diagnostics
BASE=$(mktemp -d)/metrics.cpp
BASEH=$(mktemp -d)/metrics.h

cd "$REPO" || exit 1

cp "$SRC" "$BASE"
cp "$HDR" "$BASEH"

# --- baseline must be pristine -------------------------------------------------
if ! grep -q 'if (probs\[i\] >= threshold) keep.push_back(i);' "$BASE"; then
  echo "ABORT: baseline is NOT pristine (threshold line missing)"; exit 1
fi
if ! grep -q 'const size_t per_bin = (n + n_bins - 1) / n_bins;' "$BASE"; then
  echo "ABORT: baseline is NOT pristine (ceil split line missing)"; exit 1
fi
if ! grep -q 'return total / static_cast<double>(nonempty);' "$BASE"; then
  echo "ABORT: baseline is NOT pristine (uniform weight line missing)"; exit 1
fi
echo "baseline verified pristine"
echo

restore() {
  cp "$BASE" "$SRC"
  cp "$BASEH" "$HDR"
  # Force a rebuild on the next make. Without this the binary left behind by
  # the LAST mutation survives the restore, and the next `make` sees the
  # restored source as OLDER than that binary and skips the relink -- so a
  # green-looking suite is really running M7. Observed live: the run right
  # after the harness reported "30 passed, 11 failed" from the M7 binary
  # while the source on disk was already correct. Touching the restored
  # source makes it newer than the binary and forces the rebuild.
  touch "$SRC" "$HDR"
  make "$TARGET" >/dev/null 2>&1
}
trap restore EXIT

CAUGHT=0; SURVIVED=0; BROKEN=0; INERT=0

run_mutation() {
  local name="$1"; shift
  restore
  "$@"
  # Confirm the mutation actually changed the file.
  if diff -q "$BASE" "$SRC" >/dev/null && diff -q "$BASEH" "$HDR" >/dev/null; then
    echo "MUT-NOOP   $name  (sed did not apply -- not a real mutation)"
    BROKEN=$((BROKEN+1)); return
  fi
  local log; log=$(mktemp)
  if ! make "$TARGET" >"$log" 2>&1; then
    if grep -qi "error" "$log"; then
      echo "MUT-BROKEN $name  (failed to compile -- proves nothing)"
      BROKEN=$((BROKEN+1)); rm -f "$log"; return
    fi
  fi
  # Guard against the silent-stale-binary failure mode: make must have
  # actually relinked. Comparing the binary's mtime against the source is the
  # cheapest reliable signal.
  if [ "$SRC" -nt "./$TARGET" ]; then
    echo "MUT-BROKEN $name  (binary is older than the mutated source -- stale build)"
    BROKEN=$((BROKEN+1)); rm -f "$log"; return
  fi
  if "./$TARGET" >"$log" 2>&1; then
    echo "MUT-SURVIVED $name  <<< TESTS STILL PASS -- VACUOUS"
    SURVIVED=$((SURVIVED+1))
  else
    local n; n=$(grep -c "\[FAIL\]" "$log")
    echo "MUT-CAUGHT $name  (by $n assertion(s))"
    CAUGHT=$((CAUGHT+1))
  fi
  rm -f "$log"
}

# M1: strict `>` threshold instead of `>=`. Drops predictions exactly AT the
# threshold, so a model that predicts 1.0 with a threshold of 1.0 reports a
# perfect 0.0 score. Should be caught by test 5's boundary assertion.
run_mutation "M1 threshold > instead of >=" \
  sed -i 's|if (probs\[i\] >= threshold) keep.push_back(i);|if (probs[i] > threshold) keep.push_back(i);|' "$SRC"

# M2: population-weighted ACE instead of uniform 1/R. The defining property of
# the metric. Should be caught by tests 3, 4 and 8.
run_mutation "M2 count-weighted instead of uniform 1/R" \
  sed -i 's|return total / static_cast<double>(nonempty);|{ double wsum=0.0, wtotal=0.0; for (size_t r=0;r<n_bins;++r) wsum+=bin_size[r]; for (size_t r=0;r<n_bins;++r){ if(!bin_size[r])continue; double sp=0.0,h=0.0; for(size_t s=0;s<keep.size();++s){ if(bin_of[s]!=r)continue; sp+=kp[s]; h+=(labels[keep[s]]==pos_label)?1.0:0.0; } double c=(double)bin_size[r]; wtotal += (c/wsum)*std::fabs(h/c-sp/c);} return wtotal; }|' "$SRC"

# M3: floor instead of ceil for the equal-count split. The documented
# departure; reverts to the paper's literal floor wording. Test 4 pins the
# resulting value (0.2 vs 0.175).
run_mutation "M3 floor split instead of ceil" \
  sed -i 's|const size_t per_bin = (n + n_bins - 1) / n_bins;  // ceil(N/R), min 1|const size_t per_bin = (n / n_bins);|' "$SRC"

# M4: drop the ascending sort entirely, so the "equal-count" ranges are formed
# over INPUT order rather than by confidence. On a fixture whose rows are not
# already sorted this changes the bin membership. Valid C++ (M4 v1 flipped the
# comparator to `>` inside a lambda whose body was mutated by the shell into
# invalid syntax and failed to COMPILE -- a compile failure proves nothing
# about test strength, so it is reported as MUT-BROKEN and never counted).
run_mutation "M4 no sort (ranges formed in input order)" \
  sed -i '897s|return probs\[a\] < probs\[b\];|return a < b;|' "$SRC"

# M5: classwise per-class weight |B|/n -> uniform. Makes classwise-ECE
# identical to ACE and destroys the distinction the two papers draw.
run_mutation "M5 classwise uniform weight instead of |B|/n" \
  sed -i 's|const double w = static_cast<double>(c.counts\[b\]) / n;|const double w = 1.0 / static_cast<double>(c.counts.size());|' "$SRC"

# M6: drop the per-class mean entirely (return the first class only).
run_mutation "M6 classwise ignores classes 1..k" \
  sed -i 's|return sum / static_cast<double>(per.size());|return per.empty() ? 0.0 : per[0];|' "$SRC"

# M7: threshold comparison silently inverted (keep only BELOW threshold).
run_mutation "M7 keep below threshold instead of at/above" \
  sed -i 's|if (probs\[i\] >= threshold) keep.push_back(i);|if (probs[i] <= threshold) keep.push_back(i);|' "$SRC"

echo
echo "=== SUMMARY: caught=$CAUGHT survived=$SURVIVED broken=$BROKEN ==="
[ "$SURVIVED" -eq 0 ] || exit 1