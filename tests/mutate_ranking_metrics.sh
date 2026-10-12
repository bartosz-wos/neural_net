#!/usr/bin/env bash
# mutate_ranking_metrics.sh — mutation harness for the ranking metrics in
# include/nn/utils/metrics.cpp (DCG / NDCG / MAP / MRR).
#
# The three mandatory defences, from the Grad-CAM, metrics and calibration
# sessions where a harness was itself the bug:
#   * Take the baseline ONCE, before any mutation, and ASSERT it is pristine
#     by grepping for lines that exist only in the correct implementation. A
#     harness that snapshots while a mutation is applied restores a MUTANT over
#     correct code and everything after it looks green for the wrong reason.
#   * Verify each sed ACTUALLY APPLIED (diff -q) before building, and report a
#     no-op as MUT-NOOP -- never as "survived".
#   * Report compile failure as MUT-BROKEN, never as MUT-CAUGHT. A mutant that
#     fails to build proves nothing about test strength.
# Plus: a mtime check, because `make /abs/path` reports "up to date" and skips
# the rebuild, which makes every mutation run against a STALE binary.
set -u

REPO=/home/stefan/neural_net
SRC="$REPO/include/nn/utils/metrics.cpp"
HDR="$REPO/include/nn/utils/metrics.h"
# RELATIVE target with the repo as cwd: an ABSOLUTE target silently does
# nothing (make reports "up to date"), so every mutant would run unbuilt.
TARGET=build/test_ranking_metrics
TMP=$(mktemp -d)
BASE="$TMP/metrics.cpp"
BASEH="$TMP/metrics.h"

cd "$REPO" || exit 1

cp "$SRC" "$BASE"
cp "$HDR" "$BASEH"

# --- the baseline must be pristine -------------------------------------------
# Each of these is a line that exists ONLY in the correct implementation, so a
# grep miss means the snapshot captured something other than what we intend.
for needle in \
  'for (size_t t = i; t < j; ++t) dsum += discount[t];' \
  'if (rel[d] > 0.0) order[w++] = d;' \
  'out[q] = ideal == 0.0 ? 0.0 : gain / ideal;' \
  'const size_t window = (k == 0 || k > n) ? n : k;' \
  'if (k != 0 && j >= k) break;' ; do
  if ! grep -qF "$needle" "$BASE"; then
    echo "ABORT: baseline is NOT pristine (missing: $needle)"; exit 1
  fi
done
if ! grep -qF 'double mean_reciprocal_rank' "$BASEH"; then
  echo "ABORT: baseline header is NOT pristine (mean_reciprocal_rank missing)"; exit 1
fi
echo "baseline verified pristine (5 source needles + 1 header needle)"
echo

restore() {
  cp "$BASE" "$SRC"
  cp "$BASEH" "$HDR"
  # Touch so the restored source is NEWER than the last mutant's binary;
  # otherwise make sees a newer binary and skips the relink, and the "green"
  # suite is really still running the previous mutant.
  touch "$SRC" "$HDR"
  make "$TARGET" >/dev/null 2>&1
}
trap restore EXIT

CAUGHT=0; SURVIVED=0; BROKEN=0

run_mutation() {
  local name="$1"; shift
  restore
  "$@"
  if diff -q "$BASE" "$SRC" >/dev/null && diff -q "$BASEH" "$HDR" >/dev/null; then
    echo "MUT-NOOP   $name  (sed did not apply -- not a real mutation)"
    BROKEN=$((BROKEN+1)); return
  fi
  local log; log=$(mktemp)
  if ! make "$TARGET" >"$log" 2>&1; then
    echo "MUT-BROKEN $name  (failed to compile -- proves nothing about tests)"
    BROKEN=$((BROKEN+1)); rm -f "$log"; return
  fi
  if [ "$SRC" -nt "./$TARGET" ]; then
    echo "MUT-BROKEN $name  (binary older than the mutated source -- stale build)"
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

# M1: THE BUG THIS SUITE WAS WRITTEN AROUND. Index the discount by DOCUMENT id
# instead of by RANK, which makes the DCG invariant to the ranking. This is the
# exact defect the ascending-score fixtures caught.
run_mutation "M1 discount indexed by document id, not rank" \
  sed -i 's|for (size_t t = i; t < j; ++t) dsum += discount\[t\];|for (size_t t = i; t < j; ++t) dsum += discount[order[t]];|' "$SRC"

# M2: drop the `k` truncation of the discount. Turns every @k metric into the
# full-length one; test 3 and test 10 exist for this.
run_mutation "M2 ignore k (no discount truncation)" \
  sed -i 's|if (k != 0 \&\& j >= k) break;|if (false) break;|' "$SRC"

# M3: off-by-one in the discount base -- the classic 1/log2(rank) starting at
# j+1 instead of j+2, which divides by log(1) = 0 at the top rank.
run_mutation "M3 discount base j+1 instead of j+2" \
  sed -i 's|d\[j\] = lb / std::log(static_cast<double>(j) + 2.0);|d[j] = lb / std::log(static_cast<double>(j) + 1.0);|' "$SRC"

# M4: NORMALISE BY THE FULL-LENGTH IDEAL even when k truncates. The IDCG is
# the rule test 5 exists to pin. The `+2.0` here is the SAME discount the gain
# uses; a build of the discount with NO k cutoff is what the mutation needs,
# so instead the ideal is taken from a freshly-computed untruncated discount
# inside the loop.
run_mutation "M4 ndcg ideal ignores the k truncation" \
  sed -i 's|for (size_t j = 0; j < rel.cols; ++j) ideal += by_rel\[j\] \* discount\[j\];|for (size_t j = 0; j < rel.cols; ++j) ideal += by_rel[j] * (std::log(log_base) / std::log(static_cast<double>(j) + 2.0));|' "$SRC"

# M5: the all_irrelevant branch divides anyway. Produces 0/0 -> NaN instead of
# the documented 0.0, and the fixture that returns 0.0 for everything is the
# fingerprint.
run_mutation "M5 divide by zero ideal (no all_irrelevant branch)" \
  sed -i 's|out\[q\] = ideal == 0.0 ? 0.0 : gain / ideal;|out[q] = gain / ideal;|' "$SRC"

# M6: tie-AVERAGE replaced by pessimistic in dcg_row -- i.e. take the FIRST
# document's gain in a tied group instead of the mean. The DCG family's own
# tie rule, which is deliberately different from MAP/MRR's.
run_mutation "M6 dcg tie group uses first gain, not the mean" \
  sed -i 's|gain /= static_cast<double>(j - i);|gain = rel[order[i]];|' "$SRC"

# M7: MAP/MRR tie rule flipped to OPTIMISTIC (relevant first in a tied group).
# The mirror image of the shipped pessimistic rule; separates it from the
# tie-averaged rule rather than merely inverting it.
run_mutation "M7 map/mrr ties are optimistic (relevant first)" \
  sed -i 's|if (rel\[d\] <= 0.0) order\[w++\] = d;|if (rel[d] > 0.0) order[w++] = d;|' "$SRC"

# M8: MAP's `hits / rank` accumulated as `rank / hits` instead. Both terms are
# non-zero on the same fixtures, so this is a genuine structural swap rather
# than a drop -- it survives only if nothing checks the magnitude.
run_mutation "M8 map precision term inverted (rank/hits)" \
  sed -i 's|\*ap_out = hits == 0 ? 0.0 : sum_prec / static_cast<double>(hits);|*ap_out = hits == 0 ? 0.0 : static_cast<double>(window) / sum_prec;|' "$SRC"

# M9: the per-group snapshot in rank_indices_pessimistic removed -- reintroduces
# the aliasing bug the suite caught. Guards the fix itself.
run_mutation "M9 drop the tied-group snapshot (aliasing bug)" \
  sed -i 's|const size_t d = group\[t\];|const size_t d = order[i + t];|g' "$SRC"

# M10: aggregate drops the last query instead of averaging over all of them.
run_mutation "M10 aggregate averages over n-1 queries" \
  sed -i 's|return v.empty() ? 0.0 : s / static_cast<double>(v.size());|return v.empty() ? 0.0 : s / static_cast<double>(v.size() > 0 ? v.size() - 1 : 1);|' "$SRC"

echo
echo "=== SUMMARY: caught=$CAUGHT survived=$SURVIVED broken=$BROKEN ==="
[ "$SURVIVED" -eq 0 ] || exit 1