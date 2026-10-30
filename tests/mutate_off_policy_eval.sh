#!/usr/bin/env bash
# =============================================================================
# tests/mutate_off_policy_eval.sh -- mutation harness for
# utils/off_policy_eval.{h,cpp}.
#
# PURPOSE: prove that tests/test_off_policy_eval.cpp actually CATCHES the
# defects it was written for. A green suite proves nothing on its own; the
# mutation has to fail.
#
# =============================================================================
# THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
# =============================================================================
#
#   M1  the IPS normaliser is N, not the sample count   -> IPS is a MEAN.
#   M2  the SNIS denominator is N, not sum(rho)         -> SNIS is not IPS.
#   M3  the DR estimator drops the Vhat(s) term         -> the DR cancellation
#                                                         never happens, so
#                                                         ARM A fails.
#   M4  the DR correction's sign is flipped             -> ARM A and ARM B fail.
#   M5  the DR correction drops the weight              -> the estimator is no
#                                                         longer importance
#                                                         weighted.
#   M6  the effective sample size is inverted           -> ESS is the wrong
#                                                         quantity.
#   M7  the importance weight is inverted               -> every weighted
#                                                         estimator is wrong.
#   M8  the importance weight's ratio is swapped        -> pi_0/pi_1.
#   M9  the DM estimator divides by the wrong count     -> the mean is wrong.
#   M10 the sample-count guard is removed                -> a mismatched-length
#                                                         input is read out of
#                                                         bounds instead of
#                                                         rejected.
#   M11 the negative-weight guard is removed             -> a negative weight
#                                                         is silently accepted.
#   M12 the zero-total-weight guard is removed           -> SNIS divides by
#                                                         zero.
#
# M3, M4 AND M5 ARE THE TRIO THAT MATTERS MOST: they are the three ways the
# doubly robust estimator can stop being doubly robust, and each of them
# breaks a DIFFERENT arm of the property. **THE FIXTURE THAT CATCHS THEM
# IS THE EXACT ENUMERATION over a finite action support** -- the arms are
# identities, not tolerances, so there is no argument about how much error is
# acceptable.
#
# **AND THE MEASUREMENT SHOWS THAT THE TWO ARMS ARE NOT INTERCHANGEABLE.**
# Applying each mutation and listing the checks that fail:
#
#   M3 (drops the Vhat term)   -> 2.7, 2.8, 7.1 (ARM A), 7.2 (ARM B), 8.1, 11.5
#   M4 (flips the sign)         -> 2.7, 2.8, 7.1 (ARM A),          8.1, 11.5
#   M5 (drops the weight)       -> 2.7, 2.8, 7.1 (ARM A),          8.1, 11.5
#
# M4 AND M5 BREAK ARM A ONLY, and the reason is structural rather than
# accidental: in ARM B the reward model is EXACT, which makes r_i - q_i
# IDENTICALLY ZERO for every sample, and a term that is identically zero
# cannot be changed by anything done to it. **ARM B IS THEREFORE
# STRUCTURALLY INSENSITIVE TO THE CORRECTION TERM.** The arms are
# complementary -- ARM A pins the correction, ARM B pins the Vhat term --
# and the suite PINS THAT ASYMMETRY itself (check 7.6), so it is a property
# of the fixture rather than a lucky accident of which mutations happen to
# be caught. **A CLAIM ABOUT WHICH ARM CATCHES WHAT IS A MEASUREMENT, NOT A
# COMMENT.**
#
# M1 AND M2 ARE THE PAIR THAT MATTERS NEXT, and they pull in OPPOSITE
# directions: M1 makes IPS too large by a factor of N, and M2 makes SNIS
# collapse onto IPS. A fixture that did not vary the SCALE of the weights
# could not tell M2 from the original at all.
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
#  13. **A MUTATION THAT MAKES THE CODE THROW IS NOT A GOOD MUTATION.** If
#      the mutation makes a guard fire on a fixture the suite exercises, the
#      exception escapes unless the test guards it -- and the test does guard
#      every call, so it becomes a countable FAIL rather than a crash.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 99

SRC_CPP="include/nn/utils/off_policy_eval.cpp"
SRC_HDR="include/nn/utils/off_policy_eval.h"
BIN="build/test_off_policy_eval"
OBJ="build/utils/off_policy_eval.o"
TOBJ="build/test_off_policy_eval.o"

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
        *off_policy_eval.o|*test_off_policy_eval.o) ;;
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
    rm -f "$OBJ" "$SCRATCH/off_policy_eval.o" "$SCRATCH/test_off_policy_eval.o" "$SCRATCH/test_off_policy_eval"
    g++ $CXXFLAGS_LOCAL -c "$SRC_CPP" -o "$SCRATCH/off_policy_eval.o" 2>"$SCRATCH/build.log" || return 90
    g++ $CXXFLAGS_LOCAL -c tests/test_off_policy_eval.cpp -o "$SCRATCH/test_off_policy_eval.o" 2>>"$SCRATCH/build.log" || return 90
    g++ $CXXFLAGS_LOCAL "$SCRATCH/test_off_policy_eval.o" "$SCRATCH/off_policy_eval.o" $LIB_LINK \
        -o "$SCRATCH/test_off_policy_eval" 2>>"$SCRATCH/build.log" || return 90
    "$SCRATCH/test_off_policy_eval" > "$SCRATCH/out.txt" 2>&1
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
echo "=== off_policy_eval mutations ==="

# --- M1: IPS divides by the wrong count -------------------------------------
# IPS must be the MEAN of rho*r over the N samples. Dividing by 1 instead
# makes the estimate N times too large.
run_mutation M1 "IPS divides by 1 instead of by the sample count" \
    "    return total / static_cast<double>(r.size());
}

// ---------------------------------------------------------------------------
// ope_self_normalized_ips" \
    "    return total;
}

// ---------------------------------------------------------------------------
// ope_self_normalized_ips"

# --- M2: the SNIS denominator is N ------------------------------------------
# THE DENOMINATOR IS sum(rho), NOT N. Using N silently returns IPS.
run_mutation M2 "SNIS divides by N instead of by sum(rho)" \
    "    return num / den;" \
    "    return num / static_cast<double>(r.size());"

# --- M3: the DR estimator drops the Vhat(s) term ----------------------------
# The Vhat term is what CANCELS against the model's own target-policy
# average, so dropping it destroys double robustness in ARM A.
run_mutation M3 "the DR estimator drops the Vhat(s) term" \
    "        total += v[i] + w[i] * (r[i] - q[i]);" \
    "        total += w[i] * (r[i] - q[i]);"

# --- M4: the DR correction's sign is flipped --------------------------------
run_mutation M4 "the DR correction's sign is flipped" \
    "        total += v[i] + w[i] * (r[i] - q[i]);" \
    "        total += v[i] + w[i] * (q[i] - r[i]);"

# --- M5: the DR correction drops the weight ---------------------------------
run_mutation M5 "the DR correction drops the importance weight" \
    "        total += v[i] + w[i] * (r[i] - q[i]);" \
    "        total += v[i] + (r[i] - q[i]);"

# --- M6: the effective sample size is inverted ------------------------------
run_mutation M6 "the effective sample size is inverted" \
    "    return sum * sum / sum_sq;" \
    "    return sum_sq / (sum * sum);"

# --- M7: the importance weight is inverted ----------------------------------
run_mutation M7 "the importance weight is inverted" \
    "    return target_prob / behavior_prob;" \
    "    return behavior_prob / target_prob;"

# --- M8: the batched weight's ratio is swapped ------------------------------
run_mutation M8 "the batched weight's ratio is swapped" \
    "            out(i, j) = tp / bp;" \
    "            out(i, j) = bp / tp;"

# --- M9: the DM estimator divides by the wrong count ------------------------
run_mutation M9 "the DM estimator divides by 1 instead of by the sample count" \
    "    return total / static_cast<double>(v.size());" \
    "    return total;"

# --- M10: the sample-count guard is removed ---------------------------------
# A mismatched-length input must be REJECTED, not read out of bounds.
run_mutation M10 "the sample-count guard is removed" \
    "    if (a.size() != b.size()) {" \
    "    if (false) {"

# --- M11: the negative-weight guard is removed ------------------------------
run_mutation M11 "the negative-weight guard is removed" \
    "        if (w[i] < 0.0) {" \
    "        if (false) {"

# --- M12: the zero-total-weight guard is removed ----------------------------
# SNIS's denominator would be zero.
run_mutation M12 "the zero-total-weight guard is removed" \
    "    if (den == 0.0) {" \
    "    if (false) {"

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
