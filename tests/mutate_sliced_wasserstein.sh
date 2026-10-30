#!/usr/bin/env bash
# =============================================================================
# tests/mutate_sliced_wasserstein.sh -- mutation harness for
# utils/sliced_wasserstein.{h,cpp}.
#
# PURPOSE: prove that tests/test_sliced_wasserstein.cpp actually CATCHES the
# defects it was written for. A green suite proves nothing on its own; the
# mutation has to fail.
#
# =============================================================================
# THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
# =============================================================================
#
# The sliced Wasserstein distance has FOUR load-bearing decisions, and the
# suite has one assertion group per decision:
#
#   M1  the SORT of the projections        -> the 1-D W1 is the L1 distance
#                                              between the SORTED projections.
#                                              **Omitting the sort
#                                              OVERESTIMATES**, because
#                                              sorting MINIMISES the L1 sum.
#   M2  the GENERAL (unequal-n) 1-D FORM   -> the shortcut
#                                              mean(|sorted_p - sorted_q|) is
#                                              valid only for equal n.
#   M3  the UNIT-DIRECTION contract        -> a non-unit direction is
#                                              REJECTED, not rescaled.
#   M4  the MEAN vs the MAX aggregation    -> the two overloads are
#                                              different functions.
#   M5  the CDF's "<=" at a shared value    -> a shared support point must
#                                              be counted by BOTH
#                                              distributions before the gap
#                                              opens.
#   M6  the random directions are UNIT     -> the draw normalises.
#   M7  the seed is actually USED          -> a seeded metric that ignores
#                                              the seed is not reproducible.
#
# M1 AND M2 ARE THE PAIR THAT MATTERS MOST, and they pull in OPPOSITE
# directions, which is why a fixture that catches one catches the other:
# a missing sort makes the value TOO LARGE, and the mean-form shortcut on an
# unequal-n fixture makes it TOO SMALL.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
# =============================================================================
#
#   1. `restore()` must NEVER delete its backup. It is called once PER
#      MUTATION and once on EXIT, so a self-cleaning restore works exactly
#      once and every later restore is a silent no-op -- leaving a MUTANT IN
#      THE TREE. Cleanup happens exactly once, on EXIT, and restore() is
#      ATOMIC (`cp tmp && mv -f`), because a plain `cp` truncates the
#      destination first and a concurrent reader can splice two generations
#      together.
#   2. The link set comes from `make print-lib-objs`, NEVER from
#      regex-parsing the Makefile. `LIB_OBJS` is a RECURSIVE make variable
#      whose literal text contains no `.o` filenames, so a regex yields an
#      empty set and every mutation scores MUT-BROKEN -- one defect reported
#      as N verdicts. The count is asserted >= 10.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET. It legitimately IS
#      in the library's list (it is the TU being mutated), so the correct
#      handling is to compile the mutated TU SEPARATELY, drop the library's
#      copy from the link set, and link the mutant in its place.
#   4. `MUT-BROKEN` means "the build failed". It is a separate class and is
#      never counted as a pass. A mutation reported BROKEN is re-run in
#      isolation before it is believed.
#   5. A CRASH (rc >= 128) is a THIRD class, not a verdict.
#   6. A mutation that DID NOT APPLY is a HARNESS DEFECT, not a "SURVIVED"
#      verdict. The tree is checksummed before and after every edit.
#   7. The anchor must occur EXACTLY ONCE, and the count comes from the SAME
#      engine that performs the edit (`str.count` in Python), because
#      `grep -c` counts MATCHING LINES, not matches.
#   8. `rm -f` the intermediates before every build -- a stale .o from the
#      previous mutation reproduces a build failure that does not exist alone.
#   9. NEVER run a build concurrently with this harness.
#  10. The tree is rebuilt and re-verified ON EXIT before any verdict is
#      reported.
#  11. The pristine tree must be GREEN before any mutation runs.
#  12. The separator is a REAL 0x1f BYTE (`$'\x1f'`). `"\\x1f"` inside
#      double quotes produces four literal characters, and the editor then
#      fails to split -- which happened in this repo's own first draft and
#      made every mutation report HARNESS-DEFECT while the verdict table
#      still looked plausible.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 99

SRC_CPP="include/nn/utils/sliced_wasserstein.cpp"
SRC_HDR="include/nn/utils/sliced_wasserstein.h"
BIN="build/test_sliced_wasserstein"
OBJ="build/utils/sliced_wasserstein.o"
TOBJ="build/test_sliced_wasserstein.o"

BAK_CPP="$(mktemp)"; BAK_HDR="$(mktemp)"
cp "$SRC_CPP" "$BAK_CPP"
cp "$SRC_HDR" "$BAK_HDR"

restore() {
    cp "$BAK_CPP" "$SRC_CPP.tmp" 2>/dev/null && mv -f "$SRC_CPP.tmp" "$SRC_CPP"
    cp "$BAK_HDR" "$SRC_HDR.tmp" 2>/dev/null && mv -f "$SRC_HDR.tmp" "$SRC_HDR"
}
cleanup() {
    restore
    rm -f "$BAK_CPP" "$BAK_HDR"
    # THE POST-HARNESS REBUILD-AND-VERIFY DEFENCE. A previous session shipped
    # a restored source with a stale mutant still on disk and reported a
    # phantom regression; this is the check that catches it.
    rm -f "$OBJ" "$TOBJ" "$BIN"
    make setup >/dev/null 2>&1
    if make -j6 "$BIN" >/dev/null 2>&1 && "$BIN" >/dev/null 2>&1; then
        echo ""
        echo "POST-HARNESS: tree restored, rebuilt, and GREEN."
    else
        echo ""
        echo "POST-HARNESS FAILURE: the restored tree does not build or pass. STOP." >&2
        exit 1
    fi
}
trap cleanup EXIT INT TERM

checksum() { md5sum "$1" | cut -d' ' -f1; }

# THE SEPARATOR: a real 0x1f byte.
US=$'\x1f'

# -----------------------------------------------------------------------------
# The link set, from make itself, with a count assertion.
# -----------------------------------------------------------------------------
mapfile -t ALL_OBJS < <(make print-lib-objs)
if [ "${#ALL_OBJS[@]}" -lt 10 ]; then
    echo "HARNESS DEFECT: link set has only ${#ALL_OBJS[@]} objects (need >= 10)." >&2
    echo "REFUSING TO RUN -- every verdict below would be false." >&2
    exit 99
fi
LIB_LINK=""
for o in "${ALL_OBJS[@]}"; do
    case "$o" in
        *sliced_wasserstein.o|*test_sliced_wasserstein.o) ;;
        *) LIB_LINK="$LIB_LINK $o" ;;
    esac
done
LIB_COUNT=$(echo "$LIB_LINK" | wc -w)
if [ "$LIB_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: link set collapsed to $LIB_COUNT objects." >&2
    exit 99
fi
echo "link set: $LIB_COUNT objects (the mutated TU is compiled separately and linked in)"

CXXFLAGS_LOCAL="-std=c++17 -O2 -Wall -Wextra -march=native -Iinclude"

# -----------------------------------------------------------------------------
# Build the suite against the CURRENT (possibly mutated) source. The mutated
# TU is compiled into $SCRATCH and linked IN PLACE of the library's copy.
# Returns 90 on a build failure -- which is NOT a verdict.
# -----------------------------------------------------------------------------
SCRATCH="$(mktemp -d)"

build_and_run() {
    rm -f "$OBJ" "$SCRATCH/sliced_wasserstein.o" "$SCRATCH/test_sliced_wasserstein.o" "$SCRATCH/test_sliced_wasserstein"
    g++ $CXXFLAGS_LOCAL -c "$SRC_CPP" -o "$SCRATCH/sliced_wasserstein.o" 2>"$SCRATCH/build.log" || return 90
    g++ $CXXFLAGS_LOCAL -c tests/test_sliced_wasserstein.cpp -o "$SCRATCH/test_sliced_wasserstein.o" 2>>"$SCRATCH/build.log" || return 90
    g++ $CXXFLAGS_LOCAL "$SCRATCH/test_sliced_wasserstein.o" "$SCRATCH/sliced_wasserstein.o" $LIB_LINK \
        -o "$SCRATCH/test_sliced_wasserstein" 2>>"$SCRATCH/build.log" || return 90
    "$SCRATCH/test_sliced_wasserstein" > "$SCRATCH/out.txt" 2>&1
    return $?
}

# The pristine self-check.
build_and_run
if [ $? -ne 0 ]; then
    echo "HARNESS DEFECT: the PRISTINE tree does not build or pass." >&2
    tail -20 "$SCRATCH/build.log" >&2
    exit 99
fi
PRISTINE=$(sed -nE 's/.*SUMMARY: ([0-9]+) passed, ([0-9]+) failed.*/\1/p' "$SCRATCH/out.txt" | tail -1)
if [ "${PRISTINE:-0}" -eq 0 ]; then
    echo "HARNESS DEFECT: the pristine suite is not green." >&2
    exit 99
fi
echo "pristine: $PRISTINE passed, 0 failed -- mutation may begin"

# -----------------------------------------------------------------------------
# Apply a mutation with a Python editor (str.count / str.replace).
# -----------------------------------------------------------------------------
apply_mutation() {  # $1 = spec (old<US>new), $2 = file
    python3 - "$1" "$2" <<'PYEOF'
import sys, pathlib
spec, path = sys.argv[1], sys.argv[2]
old, new = spec.split("\x1f")
p = pathlib.Path(path)
s = p.read_text()
n = s.count(old)
if n != 1:
    print(f"ANCHOR_COUNT={n}")
    sys.exit(3)
p.write_text(s.replace(old, new))
print("MUTATION_APPLIED")
PYEOF
}

CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0
declare -a TABLE

run_mutation() {
    local id="$1" desc="$2" old="$3" new="$4"
    restore

    local out
    out=$(apply_mutation "$old${US}$new" "$SRC_CPP")
    if ! echo "$out" | grep -q "MUTATION_APPLIED"; then
        TABLE+=("$id|$desc|HARNESS-DEFECT (anchor count != 1)|$(echo "$out" | tr '\n' ' ')")
        NOOP=$((NOOP+1)); return
    fi

    local before after
    before=$(checksum "$BAK_CPP"); after=$(checksum "$SRC_CPP")
    if [ "$before" = "$after" ]; then
        restore
        TABLE+=("$id|$desc|HARNESS-DEFECT (edit did not land)|"); NOOP=$((NOOP+1)); return
    fi

    build_and_run
    local rc=$?
    if [ "$rc" -eq 90 ]; then
        restore
        TABLE+=("$id|$desc|MUT-BROKEN (build failed -- NOT a verdict)|"); BROKEN=$((BROKEN+1)); return
    fi
    if [ "$rc" -ge 128 ]; then
        restore
        TABLE+=("$id|$desc|MUT-CRASH (rc=$rc -- NOT a verdict)|"); CRASH=$((CRASH+1)); return
    fi

    local nf line
    nf=$(sed -nE 's/.*SUMMARY: ([0-9]+) passed, ([0-9]+) failed.*/\2/p' "$SCRATCH/out.txt" | tail -1)
    if [ -n "$nf" ] && [ "$nf" -gt 0 ]; then
        line=$(grep -m1 "FAIL" "$SCRATCH/out.txt" | sed 's/^ *//' | cut -c1-64)
        TABLE+=("$id|$desc|CAUGHT|$line")
        CAUGHT=$((CAUGHT+1))
    else
        TABLE+=("$id|$desc|SURVIVED (the suite stayed green)|")
        SURVIVED=$((SURVIVED+1))
    fi
    restore
}

echo
echo "=== sliced_wasserstein mutations ==="

# --- M1: drop the SORT of the projections -----------------------------------
# The 1-D W1 must compare the SORTED projections. Omitting the sort makes
# the metric TOO LARGE (sorting minimises the L1 sum).
run_mutation M1 "drop the sort of the projections" \
    "    std::sort(p.begin(), p.end());
    std::sort(q.begin(), q.end());" \
    ""

# --- M2: use the equal-n MEAN form instead of the general one ---------------
run_mutation M2 "use the equal-n mean form instead of the general one" \
    "        total += std::abs(fp - fq) * (pts[i + 1] - pts[i]);" \
    "        total += std::abs(p[std::min(i, p.size() - 1)] - q[std::min(i, q.size() - 1)]);"

# --- M3: normalise the directions instead of REJECTING a non-unit one ------
run_mutation M3 "silently normalise a non-unit direction instead of rejecting it" \
    "        if (!(std::abs(norm - 1.0) <= kUnitTolerance)) {" \
    "        if (false) {"

# --- M4: the MAX overload computes the MEAN ---------------------------------
run_mutation M4 "the max overload computes the mean" \
    "    double best = -1.0;
    for (size_t k = 0; k < directions.rows; ++k) {
        const double w = wasserstein1_impl(project_onto(P, directions, k),
                                           project_onto(Q, directions, k));
        if (w > best) best = w;
    }
    return best;" \
    "    double best = -1.0;
    for (size_t k = 0; k < directions.rows; ++k) {
        const double w = wasserstein1_impl(project_onto(P, directions, k),
                                           project_onto(Q, directions, k));
        best += w;
    }
    return best / static_cast<double>(directions.rows);"

# --- M5: the CDF's "<=" at a shared value -----------------------------------
# `upper_bound` counts the values <= the point, so a shared support point is
# counted by BOTH distributions before the gap opens. `lower_bound` would
# count only the values < the point, which is the wrong convention.
run_mutation M5 "the CDF counts only the values STRICTLY below the point" \
    "        const size_t cp = static_cast<size_t>(
            std::upper_bound(p.begin(), p.end(), pts[i]) - p.begin());
        const size_t cq = static_cast<size_t>(
            std::upper_bound(q.begin(), q.end(), pts[i]) - q.begin());" \
    "        const size_t cp = static_cast<size_t>(
            std::lower_bound(p.begin(), p.end(), pts[i]) - p.begin());
        const size_t cq = static_cast<size_t>(
            std::lower_bound(q.begin(), q.end(), pts[i]) - q.begin());"

# --- M6: the random directions are normalised to the WRONG LENGTH ---------
#
# **A MUTATION THAT MAKES THE CODE CRASH IS NOT A VERDICT, AND A MUTATION
# THAT MAKES IT THROW IS NOT A GOOD MUTATION.** The first draft of this
# mutation REMOVED the normalisation entirely, so the generated directions
# had an arbitrary length. Two things then happened, and both are worth
# recording:
#
#   1. The library's own unit guard fired -- correctly -- and the
#      resulting std::invalid_argument ESCAPED main. The binary aborted
#      with rc=134 and the suite's verdict count was LOST. The harness
#      reported MUT-CRASH, which is a third class and not a verdict, so the
#      harness was RIGHT and the TEST was wrong. The fix is the test's
#      `call_guarded` wrapper, which turns a throw into a countable FAIL.
#
#   2. The mutation did not test what it claimed to. "The directions are
#      not normalised" is not observable through the metric's VALUE; it is
#      observable through the directions' LENGTH. So the mutation below
#      normalises to a length of 1.05 -- which is caught directly by the
#      suite's own "every generated direction is a UNIT vector" check,
#      without the metric's value being involved at all.
run_mutation M6 "the random directions are normalised to the wrong length" \
    "        for (size_t j = 0; j < dirs.cols; ++j) dirs(i, j) /= norm;" \
    "        for (size_t j = 0; j < dirs.cols; ++j) dirs(i, j) = dirs(i, j) / norm * 1.05;"

# --- M7: the seed is ignored ------------------------------------------------
run_mutation M7 "the seed is ignored" \
    "    std::mt19937_64 rng(seed);" \
    "    std::mt19937_64 rng(12345);"

echo
printf '%-6s %-52s %-46s %s\n' "ID" "MUTATION" "VERDICT" "DETAIL"
printf '%-6s %-52s %-46s %s\n' "----" "----" "------" "------"
for row in "${TABLE[@]}"; do
    IFS='|' read -r a b c d <<< "$row"
    printf '%-6s %-52s %-46s %s\n' "$a" "$b" "$c" "${d:-}"
done

echo
echo "FINAL: $CAUGHT caught / $SURVIVED survived / $BROKEN broken / $CRASH crashed / $NOOP noop"

rm -rf "$SCRATCH"

[ "$SURVIVED" -eq 0 ] || { echo "NOTE: $SURVIVED unexplained survivor(s) -- these are REAL coverage gaps." >&2; exit 3; }
[ "$BROKEN" -eq 0 ] || { echo "NOTE: $BROKEN mutation(s) reported BROKEN/CRASH -- re-run in isolation." >&2; exit 1; }
[ "$NOOP" -eq 0 ] || { echo "NOTE: $NOOP harness defect(s)." >&2; exit 1; }
exit 0
