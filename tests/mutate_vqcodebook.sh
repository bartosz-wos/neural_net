#!/usr/bin/env bash
# mutate_vqcodebook.sh — mutation harness for utils/vqcodebook.{h,cpp}
#
# This repo has recorded NINE mutation-harness defects, including an INVERTED
# verdict, a sed that never applied, a `sed -i inert file` that corrupted the
# source it was testing, an ABSOLUTE make target that silently reported "up to
# date" and scored 7 real mutations as survived, an EMPTY LIB_OBJS link set
# that scored 8 real mutations as "did not link", and a C++ abort scored as a
# PASS because the crash pattern only listed `Aborted` -- a *bash* message a
# C++ program never prints.
#
# Every defence below is mandatory. They are numbered so a future edit that
# drops one is visible.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BIN=build/test_vqcodebook
SRC_CPP=include/nn/utils/vqcodebook.cpp
SRC_H=include/nn/utils/vqcodebook.h

pass=0; caught=0; survived=0; broken=0; crash=0; noop=0
declare -a RESULTS

# ---------------------------------------------------------------------------
# DEFENCE 1: take the baseline ONCE, before any mutation, and assert it is
# PRISTINE by grepping for lines that exist ONLY in the correct implementation.
# A harness that snapshots after mutating will eventually restore a mutant over
# correct code (Grad-CAM session).
# ---------------------------------------------------------------------------
BASE_CPP="$(mktemp)"; BASE_H="$(mktemp)"
cp "$SRC_CPP" "$BASE_CPP"; cp "$SRC_H" "$BASE_H"
trap 'cp -f "$BASE_CPP" "$SRC_CPP" 2>/dev/null; cp -f "$BASE_H" "$SRC_H" 2>/dev/null; rm -f "$BASE_CPP" "$BASE_H"' EXIT

pristine_check() {
  local missing=0
  # Each marker is a line that exists only in the CORRECT implementation.
  grep -qF 'KMeans km(num_codes' "$SRC_CPP" || { echo "  !! marker missing: KMeans ctor"; missing=1; }
  grep -qF 'out(d, k) = centers(k, d);' "$SRC_CPP" || { echo "  !! marker missing: the (D,K) transpose"; missing=1; }
  grep -qF 'return (1.0 - beta) * 0.5 * s /' "$SRC_CPP" || { echo "  !! marker missing: the 1/2 in term1"; missing=1; }
  grep -qF 'return beta * 0.5 * s /' "$SRC_CPP" || { echo "  !! marker missing: the 1/2 in term2"; missing=1; }
  grep -qF 'g(i, d) = (1.0 - beta) * (z(i, d) - z_q(i, d)) / n;' "$SRC_CPP" || { echo "  !! marker missing: g_ze factor"; missing=1; }
  grep -qF 'g(i, d) = beta * (z_q(i, d) - z(i, d)) / n;' "$SRC_CPP" || { echo "  !! marker missing: g_codebook factor"; missing=1; }
  grep -qF 'if (err[a] != err[b]) return err[a] > err[b];' "$SRC_CPP" || { echo "  !! marker missing: descending order"; missing=1; }
  return $missing
}

if ! pristine_check; then
  echo "FATAL: baseline is NOT pristine -- refusing to run."
  exit 2
fi
echo "Baseline pristine (8 markers verified)."

# ---------------------------------------------------------------------------
# DEFENCE 2 (the defect that fired on its FIRST run, again): assert the link set
# is NON-EMPTY.
#
# The VQ-VAE session's harness used the regex '^LIB_OBJS\s*=' against a Makefile
# that writes 'LIB_OBJS := ...' -- a space before the ':=' -- so the grep matched
# nothing, the link set came out EMPTY, and all eight real mutations were scored
# "MUT-BROKEN (did not link)". ONE harness defect was reported as EIGHT mutation
# verdicts, which is worse than no harness: it would have certified a suite that
# catches nothing.
#
# THIS SCRIPT REPEATED THE CLASS, in a subtler form, on its own first run. The
# naive extraction DID match the line -- but `LIB_OBJS := $(LIB_SRCS:...)` is a
# RECURSIVE make variable, so its literal text contains a make reference, not a
# list of .o files. Counting `\.o$` in it therefore yields ZERO. The fix is to
# ask MAKE for the expanded value (`make print-lib-objs`) rather than to keep
# re-parsing make syntax with a regex.
#
# The count is asserted, because an empty link set silently turns every verdict
# into MUT-BROKEN.
# ---------------------------------------------------------------------------
print_lib_objs() {
  make -f Makefile --no-print-directory print-lib-objs 2>/dev/null
}
LIB_OBJS="$(print_lib_objs)"
OBJ_COUNT=0
if [ -n "$LIB_OBJS" ]; then
  OBJ_COUNT=0
  while IFS= read -r _line; do
    case "$_line" in *.o) OBJ_COUNT=$((OBJ_COUNT+1)) ;; esac
  done <<< "$LIB_OBJS"
fi
echo "Link set: $OBJ_COUNT objects."
if [ "$OBJ_COUNT" -lt 10 ]; then
  echo "FATAL: link set has $OBJ_COUNT objects (< 10). Extraction failed --"
  echo "       scoring every mutation as MUT-BROKEN would be a false verdict."
  echo "       (check the Makefile's 'print-lib-objs' target)"
  exit 2
fi

# ---------------------------------------------------------------------------
# DEFENCE 3: apply a mutation, and VERIFY IT APPLIED. An edit that never
# happened is indistinguishable from a mutation that survived, so they are
# reported as different verdicts.
# ---------------------------------------------------------------------------
apply_mutation() {
  local file="$1" expr="$2"
  cp -f "$BASE_CPP" "$SRC_CPP"; cp -f "$BASE_H" "$SRC_H"
  sed -i "$expr" "$file"
  if diff -q "$SRC_CPP" "$BASE_CPP" >/dev/null 2>&1 && \
     diff -q "$SRC_H" "$BASE_H" >/dev/null 2>&1; then
    return 1   # MUT-NOOP
  fi
  return 0
}

# DEFENCE 4: restore ATOMICALLY. A plain `cp` truncates the destination first,
# so a concurrent reader can splice two generations together -- that is what
# produced the corrupt `nert` line in an earlier session.
restore() {
  cp -f "$BASE_CPP" "$SRC_CPP.mut.tmp" && mv -f "$SRC_CPP.mut.tmp" "$SRC_CPP"
  cp -f "$BASE_H"   "$SRC_H.mut.tmp"   && mv -f "$SRC_H.mut.tmp"   "$SRC_H"
}

# DEFENCE 5: stale objects. A leftover .o from the previous mutation made one
# mutant report a build failure that did not reproduce in isolation.
clean_objects() {
  rm -f build/utils/vqcodebook.o build/test_vqcodebook.o "$BIN"
}

# DEFENCE 6: build with a RELATIVE target. An absolute target reports "up to
# date" and never rebuilds -- that produced 7 false MUT-SURVIVED verdicts.
build_mutant() {
  clean_objects
  make "$BIN" > /tmp/mut_build.log 2>&1
  return $?
}

# DEFENCE 7: the mtime guard. Reject a run whose binary is older than the
# mutated source, because that means the rebuild did not happen.
mtime_ok() {
  [ -f "$BIN" ] || return 1
  [ "$BIN" -nt "$SRC_CPP" ] || return 1
  return 0
}

run_mutation() {
  local name="$1" file_tag="$2" expr="$3"
  local file
  case "$file_tag" in
    cpp) file="$SRC_CPP" ;;
    h)   file="$SRC_H" ;;
    *) echo "DEFECT: spec '$name' names an unknown file tag '$file_tag'"; exit 2 ;;
  esac

  if ! apply_mutation "$file" "$expr"; then
    RESULTS+=("MUT-NOOP   | $name | the sed never applied -- a harness defect, not a verdict")
    noop=$((noop+1)); restore; return
  fi

  if ! build_mutant; then
    RESULTS+=("MUT-BROKEN | $name | did not compile -- NOT counted as a catch")
    broken=$((broken+1)); restore; return
  fi

  if ! mtime_ok; then
    RESULTS+=("MUT-NOOP   | $name | binary is older than the mutated source; rebuild did not happen")
    noop=$((noop+1)); restore; return
  fi

  local out rc
  out="$(timeout 120 "./$BIN" 2>&1)"; rc=$?

  # DEFENCE 8: a CRASH is not a pass. C++ SIGABRT prints "terminate called
  # after throwing an instance of" and bash prints "Aborted" as a
  # JOB-CONTROL message a C++ program never emits, so an earlier harness
  # scored an aborting mutant as SURVIVED. Crashes get their own column.
  if [ $rc -ne 0 ] && ! printf '%s' "$out" | grep -q '\[FAIL\]'; then
    if [ $rc -ge 128 ] || printf '%s' "$out" | grep -qi 'terminate called\|core dumped\|Segmentation'; then
      RESULTS+=("MUT-CRASH  | $name | exited rc=$rc without a [FAIL] line -- a crash, not a pass")
      crash=$((crash+1)); restore; return
    fi
  fi

  # DEFENCE 9: count verdicts from [FAIL] LINES, never from the summary
  # string, so a reworded summary cannot silently invert the result.
  local nfail
  nfail=$(printf '%s' "$out" | grep -c '\[FAIL\]' || true)
  if [ "$nfail" -gt 0 ]; then
    RESULTS+=("MUT-CAUGHT | $name | $nfail assertion(s) failed")
    caught=$((caught+1))
  elif [ $rc -eq 0 ]; then
    RESULTS+=("MUT-SURVIVED | $name | suite green -- investigate before calling it inert")
    survived=$((survived+1))
  else
    RESULTS+=("MUT-CRASH  | $name | non-zero exit with no [FAIL] line (rc=$rc)")
    crash=$((crash+1))
  fi
  restore
}

echo
echo "=== MUTATIONS ==="

# M1: drop the 1/2 from d(.,.) in term1. Should be caught by the D2 ratio.
run_mutation "M1 drop the 1/2 in committedness term1" cpp \
  's|return (1.0 - beta) \* 0.5 \* s /|return (1.0 - beta) * 1.0 * s /|'

# M2: swap beta onto the wrong side in g_ze -- the D1 inversion. THE most
# important mutation in the file: it is invisible in every VALUE.
run_mutation "M2 beta on the wrong side of g_ze (D1 inversion)" cpp \
  's|g(i, d) = (1.0 - beta) \* (z(i, d) - z_q(i, d)) / n;|g(i, d) = beta * (z(i, d) - z_q(i, d)) / n;|'

# M3: sign flip on the codebook gradient.
run_mutation "M3 sign flip on g_codebook" cpp \
  's|g(i, d) = beta \* (z_q(i, d) - z(i, d)) / n;|g(i, d) = beta * (z(i, d) - z_q(i, d)) / n;|'

# M4: forget the (D, K) transpose -- return the centres as (K, D).
run_mutation "M4 k-means init returns (K, D), losing the transpose" cpp \
  's|out(d, k) = centers(k, d);|out(d, k) = centers(d, k);|'

# M5: LRU threshold off-by-one -- replace only when strictly EXCEEDS lifespan.
run_mutation "M5 LRU threshold > instead of >= (D6 off-by-one)" cpp \
  's|if (ages(0, k) >= lifespan) dead.push_back(k);|if (ages(0, k) > lifespan) dead.push_back(k);|'

# M6: ASCENDING error order -- takes the WORST-fitted rows first.
run_mutation "M6 ascending instead of descending error order (D7)" cpp \
  's|if (err\[a\] != err\[b\]) return err\[a\] > err\[b\];|if (err[a] != err[b]) return err[a] < err[b];|'

# M7: lru_tick forgets to reset the ages of used codes.
run_mutation "M7 lru_tick never resets used-code ages" cpp \
  's|for (size_t k : used) ages(0, k) = 0.0;   // idempotent on duplicates|for (size_t k : used) (void)k;  // idempotent on duplicates|'

# M8: drop the bound check in lru_tick -- out-of-range write (heap corruption).
run_mutation "M8 lru_tick drops the out-of-range guard" cpp \
  's|if (k >= k_count)|if (false)|'

# M9: M < K silently zero-pads instead of throwing (D8).
run_mutation "M9 k-means init pads instead of throwing on M < K" cpp \
  's|if (z.rows < num_codes)|if (false)|'

echo
for r in "${RESULTS[@]}"; do echo "$r"; done

# ---------------------------------------------------------------------------
# VERIFY THE TREE IS PRISTINE AGAIN -- a harness that leaves a mutant behind
# is worse than one that reports nothing.
# ---------------------------------------------------------------------------
restore
if ! pristine_check; then
  echo
  echo "FATAL: the tree was NOT restored to pristine. Aborting."
  exit 2
fi

echo
echo "=== HARNESS SUMMARY ==="
echo "  caught   : $caught"
echo "  survived : $survived   <- an unexplained survivor is NOT a gap; probe it"
echo "  broken   : $broken     <- never counted as a catch"
echo "  crash    : $crash      <- a crash is not a pass"
echo "  noop     : $noop       <- a sed that never applied is a harness defect"
exit 0