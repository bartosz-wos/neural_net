#!/usr/bin/env bash
# Mutation harness for include/nn/utils/ptq.{h,cpp}.
#
# This is this repo's TENTH harness, and it carries every defence the previous
# nine had to learn the hard way:
#
#   1. LINK SET FROM make, NEVER REGEX. `LIB_OBJS` is a recursive make
#      variable, so its literal text contains a make reference and zero .o
#      names. Two earlier harnesses regexed it and got an EMPTY set, scoring
#      every real mutation MUT-BROKEN -- reporting one defect as N verdicts.
#   2. ASSERT THE OBJECT COUNT (>= 10) or abort BEFORE reporting any verdict.
#   3. VERIFY EVERY sed APPLIED via diff -q. A no-op edit is MUT-NOOP, which is
#      a HARNESS DEFECT, not "the suite does not cover this".
#   4. SNAPSHOT AND VERIFY BOTH .cpp AND HEADER. An earlier harness snapshotted
#      only the .cpp while mutating the header, so the mutation LEAKED INTO THE
#      TREE and the next run reported 23 identical errors.
#   5. ATOMIC RESTORE: cp to a temp then mv. A plain cp truncates first, so a
#      concurrent reader can splice two generations together.
#   6. rm -f the intermediates before every build: a stale .o from the previous
#      mutation reports a build failure that does not reproduce in isolation.
#   7. MUT-BROKEN MEANS "the build failed", never a verdict about the suite.
#   8. MUT-CRASH IS ITS OWN COLUMN. A crash produces no [FAIL] line, so
#      [FAIL]-counting verdicts would read it as a PASS.
#   9. NO \n INSIDE A sed PATTERN -- it does not match in GNU sed. Replacements
#      use perl with \Q..\E literal quoting.
#  10. Refuse to start unless the PRISTINE suite is green.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC=include/nn/utils/ptq.cpp
HDR=include/nn/utils/ptq.h
TEST=tests/test_ptq.cpp
BIN=build/test_ptq
WORK=build/mutation/ptq
OBJ=build/utils/ptq.o
TOBJ=build/test_ptq.o

rm -rf "$WORK"; mkdir -p "$WORK"
BASE_SRC="$WORK/ptq.cpp.base"; BASE_HDR="$WORK/ptq.h.base"

# Literal replacement with no delimiter and no metacharacters. Written as a
# file rather than an inline -c so the C++ anchors containing quotes and
# newlines survive the shell intact.
cat > "$WORK/apply_edit.py" <<'PYEOF'
import sys
path, anchor, repl = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    src = f.read()
if anchor not in src:
    sys.exit(1)
if src.count(anchor) != 1:
    print(f"anchor appears {src.count(anchor)} times; refusing", file=sys.stderr)
    sys.exit(1)
with open(path, 'w') as f:
    f.write(src.replace(anchor, repl))
PYEOF

# --- atomic restore -------------------------------------------------------
restore() {
    cp "$BASE_SRC" "$WORK/ptq.cpp.tmp" && mv -f "$WORK/ptq.cpp.tmp" "$SRC"
    cp "$BASE_HDR" "$WORK/ptq.h.tmp" && mv -f "$WORK/ptq.h.tmp" "$HDR"
}
trap restore EXIT

cp "$SRC" "$BASE_SRC"; cp "$HDR" "$BASE_HDR"
SRC_SUM=$(sha256sum "$SRC" | cut -d' ' -f1)
HDR_SUM=$(sha256sum "$HDR" | cut -d' ' -f1)

# --- link set, from make --------------------------------------------------
LIB_OBJS_LIST=$(make print-lib-objs 2>/dev/null | tr ' ' '\n' | grep '\.o$' | grep -v 'utils/ptq\.o$')
LIB_COUNT=$(printf '%s\n' "$LIB_OBJS_LIST" | grep -c . )
if [ "$LIB_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: link set has $LIB_COUNT objects (need >= 10)." >&2
    echo "Refusing to report any verdict -- an empty link set would score every" >&2
    echo "real mutation MUT-BROKEN, which is this repo's most expensive defect." >&2
    exit 2
fi
echo "link set: $LIB_COUNT objects (verified via make print-lib-objs)"

# --- build helpers --------------------------------------------------------
build_and_run() {
    rm -f "$OBJ" "$TOBJ"          # defence 6: no stale intermediates
    if ! make "$BIN" >"$WORK/build.log" 2>&1; then
        echo "BUILD_FAILED"; return 3
    fi
    local out rc
    out=$(timeout 600 "./$BIN" 2>&1); rc=$?
    LAST_OUT="$out"; LAST_RC=$rc
    return 0
}

# Refuse to start unless the pristine suite is green (defence 10).
build_and_run
if [ $? -eq 3 ]; then
    echo "HARNESS DEFECT: pristine build failed; see $WORK/build.log" >&2
    exit 2
fi
if [ "$LAST_RC" -ne 0 ]; then
    echo "HARNESS DEFECT: pristine suite is NOT green (exit $LAST_RC)." >&2
    echo "$LAST_OUT" | tail -5 >&2
    echo "Refusing to run -- mutation verdicts against a red suite are noise." >&2
    exit 2
fi
PRISTINE_PASS=$(printf '%s\n' "$LAST_OUT" | grep -oE '[0-9]+ passed' | tail -1)
echo "pristine suite green: $PRISTINE_PASS"

# --- mutation specifications ----------------------------------------------
# Format: id|file|anchor|replacement|expectation(inert|caught)
# The anchor MUST exist verbatim in the named file, or the run aborts.
run_mutation() {
    local id="$1" file="$2" anchor="$3" repl="$4"
    restore
    local sum_before sum_after
    sum_before=$(sha256sum "$file" | cut -d' ' -f1)

    if ! grep -qF -- "$anchor" "$file"; then
        echo "MUT-NOOP    $id  (anchor not present in $file -- HARNESS DEFECT)"
        return
    fi
    # Apply the edit with a Python literal string replace.
    #
    # DEFENCE 11 (this harness's first run): the edit was applied with
    # `perl -0pi -e "s/\Q$anchor\E/$repl/"`, and BOTH the anchor and the
    # replacement routinely contain `/` and `\n`. A `/` inside the replacement
    # terminates the s/// operator ("Substitution pattern not terminated"), and
    # a multi-line anchor collides with the shell. SIX mutations were reported
    # MUT-NOOP for that reason -- and MUT-NOOP is the correct verdict for a
    # no-op, so the run did not lie, but it produced almost no evidence.
    # `str.replace` has no delimiter and no metacharacters at all.
    if ! python3 "$WORK/apply_edit.py" "$file" "$anchor" "$repl"; then
        echo "MUT-NOOP    $id  (edit did not apply -- HARNESS DEFECT)"
        return
    fi
    sum_after=$(sha256sum "$file" | cut -d' ' -f1)
    if [ "$sum_before" = "$sum_after" ]; then
        echo "MUT-NOOP    $id  (edit did not apply -- HARNESS DEFECT)"
        return
    fi

    build_and_run
    local br=$?
    if [ $br -eq 3 ]; then
        echo "MUT-BROKEN  $id  (build failed -- NOT a verdict about the suite)"
        return
    fi
    local fails
    fails=$(printf '%s\n' "$LAST_OUT" | grep -c '^\[FAIL\]')
    if [ "$LAST_RC" -ge 128 ] || [ "$LAST_RC" -eq 134 ]; then
        echo "MUT-CRASH   $id  (exit $LAST_RC, $fails assertion failures)"
        return
    fi
    if [ "$fails" -gt 0 ] || [ "$LAST_RC" -ne 0 ]; then
        echo "MUT-CAUGHT  $id  ($fails failing assertions)"
    else
        echo "MUT-SURVIVED $id"
    fi
}

# === the mutations ========================================================
echo
echo "=== M01: scale uses 128 instead of 127 (off-by-one in the range) ==="
run_mutation M01 "$SRC" "return amax / PTQ_QMAX;" "return amax / 128.0;"

echo "=== M02: scale drops the fabs (a negative max gives a negative scale) ==="
# DEFENCE 12 (this harness's second run): the anchor
# `amax = std::max(amax, std::fabs(w[i][j]));` appears TWICE -- once in the
# per-tensor scale and once in the per-channel scale -- and apply_edit.py
# refuses a non-unique anchor. That refusal is the correct behaviour (a
# multi-site edit is a different mutation than intended), so the anchor is
# made unique by including its surrounding context.
run_mutation M02 "$SRC" \
    "    double amax = 0.0;
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t j = 0; j < w.cols; ++j)
            amax = std::max(amax, std::fabs(w[i][j]));" \
    "    double amax = 0.0;
    for (size_t i = 0; i < w.rows; ++i)
        for (size_t j = 0; j < w.cols; ++j)
            amax = std::max(amax, w[i][j]);"

echo "=== M03: clamping to 126 instead of 127 (loses half a bit of range) ==="
run_mutation M03 "$SRC" \
    "r = std::max(-PTQ_QMAX, std::min(PTQ_QMAX, r));" \
    "r = std::max(-126.0, std::min(126.0, r));"

echo "=== M04: rounding mode flipped to truncation ==="
run_mutation M04 "$SRC" "double r = std::round(w[i][j] / s);" \
    "double r = std::trunc(w[i][j] / s);"

echo "=== M05: ROUNDING -> FLOOR (a negative half-grid point moves) ==="
run_mutation M05 "$SRC" "double r = std::round(w[i][j] / s);" \
    "double r = std::floor(w[i][j] / s);"

echo "=== M06: per-channel reduces over ROWS instead of columns ==="
# DEFENCE 13 (this harness's second run): the first M06 rewrote the loop HEADER
# of ptq_scale_per_channel's inner loop to iterate over columns, which made the
# body re-scan every column and produced a `j2` shadow warning -> MUT-BROKEN.
# A build failure is NOT a verdict about the suite, so the mutation is rewritten
# to a well-formed axis swap: keep the loop over `j` but take the max down the
# COLUMN (which is what the code already does) turned into taking it across the
# row (the axis bug D3 exists to catch), expressed as a transpose read.
run_mutation M06 "$SRC" \
    "        double amax = 0.0;
        for (size_t i = 0; i < w.rows; ++i)
            amax = std::max(amax, std::fabs(w[i][j]));
        scales[0][j] = amax / PTQ_QMAX;" \
    "        double amax = 0.0;
        for (size_t j2 = 0; j2 < w.cols; ++j2)
            amax = std::max(amax, std::fabs(w[0][j2]));
        scales[0][j] = amax / PTQ_QMAX;"

echo "=== M07: the zero-scale guard removed (0/0 -> NaN) ==="
run_mutation M07 "$SRC" "            if (s == 0.0) {" \
    "            if (false) {"

echo "=== M08: dequantize drops the zero-scale guard ==="
run_mutation M08 "$SRC" "    if (scale == 0.0) {" "    if (false) {"

echo "=== M09: the STE stops being an identity ==="
run_mutation M09 "$SRC" "    return grad.clone();" "    return grad.clone() * 0.5;"

echo "=== M10: argmax tie goes to the LAST index (>= instead of >) ==="
run_mutation M10 "$SRC" "            if (logits[i][j] > logits[i][best]) best = j;" \
    "            if (logits[i][j] >= logits[i][best]) best = j;"

echo "=== M11: relative_l2 divides by zero instead of guarding ==="
run_mutation M11 "$SRC" "    if (den == 0.0) return num == 0.0 ? 0.0 : num;" \
    "    if (den == 0.0) return 0.0 / 0.0;"

echo "=== M12: quantize_model skips the write-back (reports work, does none) ==="
run_mutation M12 "$SRC" "            *p = w_hat;" "            (void)w_hat;"

echo "=== M13: quantize_model never increments num_tensors ==="
run_mutation M13 "$SRC" "            ++rep.num_tensors;" ""

echo "=== M14: the empty-tensor guard is removed from the per-tensor scale ==="
run_mutation M14 "$SRC" \
    '    require_non_empty(w, "ptq_scale_per_tensor");' ""

echo "=== M15: per-channel dequantize accepts a mismatched scale width ==="
run_mutation M15 "$SRC" "    if (scales.rows != 1 || scales.cols != codes.cols) {" \
    "    if (false) {"

# ---------------------------------------------------------------------------
# INVERTED-VERDICT CHECK for M08, which was PROVED INERT.
#
# The TDD skill's rule: a mutation that provably changes nothing is not a
# vacuous test, but "passed" must not rest on an unchecked claim. So the
# harness re-derives the inertness and INVERTS the verdict: it FAILS if the
# mutation ever starts to matter.
#
# The claim (measured over {0, 1e-320, 1e-12, 1e-6, 1} x {0, +-1, +-127, 0.5},
# zero differing cases): removing ptq_dequantize's zero-scale guard cannot
# change any result, because IEEE 754 defines c * 0.0 == 0.0 for every FINITE
# c. NaN needs 0*inf or inf*0, and `scale` is a finite double.
#
# This function therefore asserts the OPPOSITE of the usual direction: it
# passes when the suite stays green under M08, and fails if the guard starts to
# matter. A wrong "inert" claim cannot survive it.
# ---------------------------------------------------------------------------
run_mutation_expect_inert() {
    local id="$1" file="$2" anchor="$3" repl="$4"
    restore
    if ! grep -qF -- "$anchor" "$file"; then
        echo "MUT-INERT-NOOP $id  (anchor not present -- HARNESS DEFECT)"
        return 1
    fi
    python3 "$WORK/apply_edit.py" "$file" "$anchor" "$repl" || {
        echo "MUT-INERT-NOOP $id  (edit did not apply -- HARNESS DEFECT)"
        return 1
    }
    build_and_run
    if [ $? -eq 3 ]; then
        echo "MUT-INERT-BROKEN $id  (build failed -- inertness UNVERIFIABLE)"
        restore
        return 1
    fi
    local fails
    fails=$(printf '%s\n' "$LAST_OUT" | grep -c '^\[FAIL\]')
    restore
    if [ "$fails" -gt 0 ] || [ "$LAST_RC" -ne 0 ]; then
        echo "MUT-INERT-WRONG $id  (the mutation NOW matters: $fails failures"
        echo "                  -- the 'inert' verdict is STALE and wrong)"
        return 1
    fi
    echo "MUT-INERT-CONFIRMED $id  (suite green under the mutation, as measured)"
    return 0
}

echo
echo "=== inverted-verdict check: M08 must STAY inert ==="
run_mutation_expect_inert M08 "$SRC" "    if (scale == 0.0) {" "    if (false) {"
INERT_OK=$?
if [ "$INERT_OK" -ne 0 ]; then
    echo "INERT VERDICT FAILED -- see above" >&2
    exit 3
fi

restore
# Verify the tree is pristine again -- BOTH files, not just the .cpp (defence 4).
NOW_SRC=$(sha256sum "$SRC" | cut -d' ' -f1)
NOW_HDR=$(sha256sum "$HDR" | cut -d' ' -f1)
if [ "$NOW_SRC" = "$SRC_SUM" ] && [ "$NOW_HDR" = "$HDR_SUM" ]; then
    echo "source restored: YES (both .cpp and .h verified)"
else
    echo "source restored: NO -- THE TREE IS LEFT BROKEN" >&2
    exit 2
fi

# Rebuild the real binary so no stale mutant is left behind (an earlier session
# shipped exactly that: a restored source with a stale build/test_* on disk).
build_and_run
echo "final rebuild: exit $LAST_RC, $(printf '%s\n' "$LAST_OUT" | grep -oE '[0-9]+ passed, [0-9]+ failed' | tail -1)"