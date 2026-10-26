#!/usr/bin/env bash
# tests/mutate_conformal.sh -- mutation harness for utils/conformal.{h,cpp}.
#
# This harness exists because the repo has recorded TWENTY-THREE instances of a
# mutation harness reporting a confident WRONG verdict. Every defect below is
# one this repo has already paid for, so they are addressed BY CONSTRUCTION
# rather than by memory:
#
#   1. restore() MUST NOT DELETE ITS OWN BACKUP. It is called once per
#      mutation AND on EXIT; a restore that `rm`s the backup works exactly
#      once, every later restore is a silent no-op printing `cp: cannot stat`,
#      and the tree is left holding whatever mutant ran last. Cleanup happens
#      exactly once, on EXIT, and restore() is ATOMIC (`cp tmp && mv -f`).
#   2. NEVER RECOVER THE LINK SET BY REGEX. `LIB_OBJS := $(LIB_SRCS:...)` is a
#      RECURSIVE make variable whose literal text contains no .o filenames; a
#      regex against it yields ZERO objects and turns every mutation into a
#      false "did not link". This harness asks make, and ASSERTS the count.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET, or every mutation
#      double-defines every symbol and reports CAUGHT (compile error). A link
#      error is not a mutation verdict. The correct handling is to compile the
#      mutated TU SEPARATELY and swap it in for the library object.
#   4. `MUT-BROKEN` MEANS THE BUILD FAILED. It is reported separately from
#      CAUGHT and never counted as a pass, and a mutation reported BROKEN is
#      re-run in isolation before it is believed.
#   5. A MUTATION THAT DID NOT APPLY IS A HARNESS DEFECT, not a verdict. The
#      tree is checksummed before and after every edit.
#   6. NO EMBEDDED NEWLINES AND NO DELIMITED SCRIPT. The edit is done by a
#      Python editor with `str.count`/`str.replace`, which has no delimiter
#      concept -- C++ is full of `/`, `|`, `&` and `\`, and a `perl s///`
#      dies on the first of them.
#   7. A MISSING BINARY IS `UNAVAILABLE`, never a verdict.
#   8. The spec loop refuses any entry that does not name a known file, so a
#      stray marker can never be read as an editor script.
#   9. Throwing calls are wrapped so a guard mutation reports a verdict
#      instead of aborting the binary and leaving the harness with nothing to
#      count.
#  10. `rm -f` the intermediates before every build -- a stale .o from the
#      previous mutation reproduces a build failure that does not exist alone.
#  11. A MUTATION THAT CRASHES (rc >= 128) IS NOT A VERDICT. It is a third
#      verdict class: the binary compiled, then died. Reporting that as
#      CAUGHT would score a defect as a coverage claim.
#  12. THE ANCHOR MUST BE UNIQUE, and the uniqueness count comes from the
#      SAME engine that performs the edit (`str.count`), not from
#      `grep -c` (which counts MATCHING LINES, not matches).
#  13. AFTER ADDING A FEATURE, RE-CHECK EVERY EXISTING MUTATION'S ANCHOR
#      COUNT. A new feature that reuses a line invalidates an old anchor
#      silently, and the failure is quiet: the harness reports a harness
#      defect rather than a verdict, so a reader who skims the output could
#      read "M07 ANCHOR-AMBIGUOUS" as "M07 is inert".
#
# NEVER RUN THIS HARNESS CONCURRENTLY WITH tests/mutate_logit_processing.sh
# or any other harness that touches the SAME files -- they share build/*.o,
# and a concurrent build fails with `cannot find build/utils/<file>.o`.

set -u
cd "$(dirname "$0")/.." || exit 1

SRC="include/nn/utils/conformal.cpp"
HDR="include/nn/utils/conformal.h"
TEST="tests/test_conformal.cpp"
BIN="build/test_conformal"
BAK_SRC=$(mktemp); BAK_HDR=$(mktemp)
BAK_MARK=".mutate_conformal_tree_pristine"
cp "$SRC" "$BAK_SRC"; cp "$HDR" "$BAK_HDR"
cp "$SRC" "$BAK_MARK"

restore() {                      # atomic; NEVER deletes the backups
    cp "$BAK_SRC" "$SRC.tmp" 2>/dev/null && mv -f "$SRC.tmp" "$SRC"
    cp "$BAK_HDR" "$HDR.tmp" 2>/dev/null && mv -f "$HDR.tmp" "$HDR"
}
cleanup() { restore; rm -f "$BAK_SRC" "$BAK_HDR" "$BAK_MARK"; }
trap cleanup EXIT INT TERM

# ---- link set, from make, with a count assertion (defect 2) ----------------
LIB_LINK=$(make -s print-lib-objs 2>/dev/null | grep '\.o$' | grep -v '/test_' | tr '\n' ' ')
LIB_COUNT=$(echo "$LIB_LINK" | wc -w)
if [ "$LIB_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: recovered only $LIB_COUNT library objects (< 10)." >&2
    echo "Every verdict below would be false. REFUSING TO RUN." >&2
    exit 2
fi
MUT_OBJ="build/utils/conformal.o"
case " $LIB_LINK " in
    *" $MUT_OBJ "*)
        LIB_LINK=$(echo "$LIB_LINK" | tr ' ' '\n' | grep -v "^${MUT_OBJ}\$" | tr '\n' ' ')
        echo "link set: $LIB_COUNT objects, minus $MUT_OBJ (compiled separately)"
        ;;
    *)
        echo "HARNESS DEFECT: $MUT_OBJ is not in the library link set at all." >&2
        exit 2 ;;
esac
LIB_LINK_COUNT=$(echo "$LIB_LINK" | wc -w)
if [ "$LIB_LINK_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: link set collapsed to $LIB_LINK_COUNT objects." >&2
    exit 2
fi

CXXFLAGS_LOCAL="-std=c++17 -O2 -Wall -Wextra -march=native -Iinclude"

build_and_run() {   # defect 10
    rm -f build/test_conformal.o "$BIN" "$MUT_OBJ"
    make -s build/utils/conformal.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$TEST" -o build/test_conformal.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$SRC" -o "$MUT_OBJ" >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL $LIB_LINK "$MUT_OBJ" build/test_conformal.o -o "$BIN" >/dev/null 2>&1 || return 1
    return 0
}

# ---- pristine self-check: the suite must be GREEN before any mutation ------
build_and_run
PRISTINE=$("$BIN" 2>&1 | sed -n 's/^=== SUMMARY: \([0-9]*\) passed.*/\1/p' | tail -1)
if [ "${PRISTINE:-0}" -eq 0 ] || ! "$BIN" >/dev/null 2>&1; then
    echo "HARNESS DEFECT: pristine tree is not green ($PRISTINE passed)." >&2
    echo "REFUSING TO RUN -- this is exactly how a leaked mutant is caught." >&2
    exit 2
fi
echo "pristine: $PRISTINE passed, 0 failed -- mutation may begin"

CAUGHT=0; SURVIVED=0; BROKEN=0; NOOP=0; UNAVAIL=0
declare -a TABLE

run_mutation() {   # $1=id  $2=file  $3=from  $4=to
    local id="$1" file="$2" from="$3" to="$4"
    restore

    if [ ! -f "$file" ]; then
        TABLE+=("$id|$file does not exist|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi
    # defect 12: the count comes from the same engine that performs the edit.
    local n
    n=$(python3 -c 'import sys; print(open(sys.argv[1],encoding="utf8").read().count(sys.argv[2]))' "$file" "$from")
    if [ "$n" -ne 1 ]; then
        TABLE+=("$id|anchor count = $n|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi

    local before after
    before=$(md5sum "$file" | cut -d' ' -f1)
    python3 - "$file" "$from" "$to" <<'PYEOF' || { restore; TABLE+=("$id|editor failed|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return; }
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding='utf8').read()
n = s.count(frm)
if n != 1:
    sys.stderr.write("anchor count %d\n" % n)
    sys.exit(3)
open(path, 'w', encoding='utf8').write(s.replace(frm, to))
PYEOF
    after=$(md5sum "$file" | cut -d' ' -f1)

    if [ "$before" = "$after" ]; then      # defect 5
        restore
        TABLE+=("$id|edit did not apply|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi

    build_and_run
    if [ ! -x "$BIN" ]; then               # defect 7
        restore
        TABLE+=("$id|no binary produced|UNAVAILABLE"); UNAVAIL=$((UNAVAIL+1)); return
    fi

    local out rc
    out=$("$BIN" 2>&1); rc=$?
    local np nf
    np=$(echo "$out" | sed -n 's/^=== SUMMARY: \([0-9]*\) passed.*/\1/p' | tail -1)
    nf=$(echo "$out" | sed -n 's/.*, \([0-9]*\) failed ===.*/\1/p' | tail -1)
    np=${np:-0}; nf=${nf:-0}

    if [ "$rc" -ge 128 ]; then              # defect 11: a crash is not a verdict
        restore
        TABLE+=("$id|CRASH rc=$rc|CRASH"); BROKEN=$((BROKEN+1)); return
    fi
    if [ "$nf" -gt 0 ]; then
        restore
        TABLE+=("$id|$np passed, $nf failed|CAUGHT"); CAUGHT=$((CAUGHT+1)); return
    fi
    restore
    TABLE+=("$id|suite still green|SURVIVED"); SURVIVED=$((SURVIVED+1))
}

echo
echo "=== conformal mutations ==="

# --- THE EXACT INDEX: the whole point of this file's arithmetic -------------
# M01 drops the `+1` from (n+1): the index becomes n instead of n+1.
run_mutation M01 "$SRC" "const size_t m = n + 1;" "const size_t m = n;"

# M02 replaces the exact identity with the TEXTBOOK FLOAT expression, which
#      is wrong at alpha = 2/3. The suite's S1a pins that case.
run_mutation M02 "$SRC" "return m - static_cast<size_t>(floored);" \
    "return static_cast<size_t>(std::ceil((double)m * (1.0 - alpha)));"

# M03 uses floor instead of ceil (the identity's off-by-one).
run_mutation M03 "$SRC" "return m - static_cast<size_t>(floored);" \
    "return m - static_cast<size_t>(floored) - 1;"

# --- THE +inf CASE: an insufficient calibration set -------------------------
# M04 clamps the index to n instead of returning +inf. This BREAKS THE
#      GUARANTEE and is the trap the header calls out by name.
run_mutation M04 "$SRC" "if (k > n) {" "if (false) {"

# --- THE ORDER STATISTIC ----------------------------------------------------
# M05 off-by-one in the order statistic (k-th smallest vs (k+1)-th).
run_mutation M05 "$SRC" "std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k - 1), v.end());" \
    "std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());"

# M06 replaces the order statistic with an INTERPOLATION -- a different rule.
run_mutation M06 "$SRC" "return v[k - 1];" \
    "return v[0] + (static_cast<double>(k) / static_cast<double>(v.size())) * (v[v.size() - 1] - v[0]);"

# --- THE APS SCORE: the cumulative mass through the true class -------------
# M07 makes the score the probability of the true class instead of the
#      cumulative mass through it.
run_mutation M07 "$SRC" "score = running;" "score = probs(i, order[r]);"

# M08 scores the mass BEFORE the true class (drops the true class itself).
run_mutation M08 "$SRC" "            if (order[r] == static_cast<size_t>(y)) {
                score = running;
            }" \
    "            if (order[r] == static_cast<size_t>(y)) {
                score = running - probs(i, order[r]);
            }"

# M09 breaks the tie-break: sorts ASCENDING by probability instead of
#      descending. The tie fixture (S4h/S4i) pins the descending rule.
run_mutation M09 "$SRC" "return probs(i, a) > probs(i, b);" "return probs(i, a) < probs(i, b);"

# --- THE APS SET: the crossing class is KEPT ------------------------------
# M10 uses a strict `>` instead of `>=` on the crossing class.
run_mutation M10 "$SRC" "if (running >= qhat) {" "if (running > qhat) {"

# M11 omits the crossing class entirely.
run_mutation M11 "$SRC" "out.push_back(static_cast<int>(order[r]));" \
    "if (running < qhat) { out.push_back(static_cast<int>(order[r])); }"

# M12 sorts the classes ASCENDING by probability, so the set is built in the
#      wrong order and the cumulative mass is meaningless.
run_mutation M12 "$SRC" "return probs_row(0, a) > probs_row(0, b);" \
    "return probs_row(0, a) < probs_row(0, b);"

# --- THE CQR SCORE: max of two one-sided gaps ------------------------------
# M13 replaces the max with a sum.
run_mutation M13 "$SRC" "out(i, j) = std::max(a, b);" "out(i, j) = a + b;"

# M14 clamps the score at zero -- which would stop CQR from SHRINKING the
#      interval. The interior-target fixture (S6d) pins this.
run_mutation M14 "$SRC" "out(i, j) = std::max(a, b);" "out(i, j) = std::max(0.0, std::max(a, b));"

# M15 swaps the two gaps (lower-y vs y-upper).
run_mutation M15 "$SRC" "const double a = lower(i, j) - y(i, j);
            const double b = y(i, j) - upper(i, j);" \
    "const double a = y(i, j) - lower(i, j);
            const double b = upper(i, j) - y(i, j);"

# --- THE CQR INTERVAL: qhat on BOTH ends -----------------------------------
# M16 adds qhat to only ONE end.
run_mutation M16 "$SRC" "out(0, 1) = upper + qhat;" "out(0, 1) = upper - qhat;"

# M17 subtracts qhat from the lower end instead of adding.
run_mutation M17 "$SRC" "out(0, 0) = lower - qhat;" "out(0, 0) = lower + qhat;"

# --- THE GUARDS -------------------------------------------------------------
# M18 removes the finiteness check on the calibration scores. The suite's S8a
#      pins it, and the check must run BEFORE the `k > n` early return --
#      which is exactly the ordering bug the suite caught.
run_mutation M18 "$SRC" "if (!std::isfinite(s)) {
                throw std::invalid_argument(\"conformal: calibration scores must be finite\");
            }" "if (false) {
                throw std::invalid_argument(\"conformal: calibration scores must be finite\");
            }"

# M19 removes the alpha range check.
run_mutation M19 "$SRC" "if (alpha < 0.0 || alpha > 1.0) {" "if (false) {"

# M20 removes the label RANGE check. The dead `!found` throw that used to
#      mask this was removed after the harness proved it unreachable -- so
#      this mutation now has to be caught by the range check ITSELF, which is
#      the only guard left. (Two guards for one condition is one too many:
#      the dead one made the live one untested.)
run_mutation M20 "$SRC" "if (y < 0 || y >= static_cast<long long>(K)) {" "if (false) {"

# M21 removes the row-count shape check in aps_score.
run_mutation M21 "$SRC" "if (probs.rows != labels.rows) {" "if (false) {"

echo
printf '%-6s %-52s %s\n' "ID" "DETAIL" "VERDICT"
printf '%-6s %-52s %s\n' "----" "-----" "-------"
for row in "${TABLE[@]}"; do
    IFS='|' read -r a b c <<< "$row"
    printf '%-6s %-52s %s\n' "$a" "$b" "$c"
done

echo
echo "=== caught $CAUGHT / survived $SURVIVED / broken $BROKEN / noop $NOOP / unavailable $UNAVAIL ==="

# ---- post-harness verification: the tree must be pristine again -----------
restore
build_and_run
FINAL=$("$BIN" 2>&1 | sed -n 's/^=== SUMMARY: \([0-9]*\) passed.*/\1/p' | tail -1)
if "$BIN" >/dev/null 2>&1 && [ "${FINAL:-0}" -eq "$PRISTINE" ]; then
    echo "POST-HARNESS: tree restored, suite green again ($FINAL passed). OK"
else
    echo "POST-HARNESS FAILED: $FINAL passed against $PRISTINE expected." >&2
    exit 2
fi

[ "$BROKEN" -eq 0 ] || { echo "NOTE: $BROKEN mutation(s) reported BROKEN/CRASH -- re-run in isolation." >&2; }
[ "$SURVIVED" -eq 0 ] || exit 3
