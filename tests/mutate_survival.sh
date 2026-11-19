#!/usr/bin/env bash
# =============================================================================
# tests/mutate_survival.sh -- mutation harness for the survival-analysis
# utilities: utils/survival.{h,cpp}.
#
# WHY THIS FILE EXISTS
# =============================================================================
#
# A suite is, by construction, unable to report a coverage gap: it can only
# report what it asserts. This harness breaks the implementation in specific
# ways and requires that the suite FAIL each time. It is the only thing in
# this repo that can find "the suite is green and the behaviour is wrong".
#
# =============================================================================
# THE MUTATIONS, AND WHAT EACH ONE IS FOR
# =============================================================================
#
#   M1  the risk set is {j : t_j > t} instead of {j : t_j >= t}
#       -> the risk-set convention (C1). The subject censored AT an event
#          time is dropped from the risk set at the moment it is known to be
#          alive. The F9 fixture exists to see this.
#   M2  the KM factor uses n_risk - 1
#       -> the Kaplan-Meier product itself.
#   M3  the survival curve is LEFT-continuous
#       -> the right-continuity convention (C2).
#   M4  the Greenwood term divides by n_risk only
#       -> Greenwood's variance (C4).
#   M5  the Greenwood term is NOT guarded against n_risk == d
#       -> the LIMIT convention (C4). Without the guard the variance is
#          inf or NaN on the everyone-dies-at-once fixture.
#   M6  the median is the first time S is STRICTLY below 0.5
#       -> the median convention (C3).
#   M7  the Nelson-Aalen hazard uses n_risk - d
#       -> the NA estimator (C5).
#   M8  the Nelson-Aalen grid is ALL distinct times
#       -> the NA grid convention (C5).
#   M9  the log-rank expected count divides by n-1
#       -> the hypergeometric expectation.
#   M10 the log-rank variance drops the finite-population factor
#       -> the hypergeometric variance. **The F5 fixture SHARES event
#          times between the groups precisely so that this factor is not
#          1** -- with d = 1 everywhere it is exactly 1 and the mutation
#          is arithmetically invisible.
#   M11 the log-rank p-value is 1 - erfc(sqrt(stat/2))
#       -> the p-value's sign. **A "p is in [0,1]" check cannot see
#          this**, which is why the suite pins the ABSOLUTE p-value.
#   M12 the log-rank statistic divides by V twice
#       -> the statistic.
#   M13 the concordance drops the half credit for tied risk scores
#       -> the tie convention (C7).
#   M14 the concordance counts the LATER failure as the correct one
#       -> the SIGN convention (C7). This is the mutation that catches the
#          two readings in the wild; the suite pins both C(s) and C(-s)
#          and their complementarity identity.
#   M15 the concordance never admits an equal-times pair
#       -> the admissibility rule (C7).
#   M16 the Cox Breslow denominator keeps the tied deaths' contribution
#       -> the tie handling (C9). **This one is the reason the suite
#          ships an INDEPENDENT SCALAR-LOOP REFERENCE**: the mutation
#          moves the value and the gradient CONSISTENTLY, so the
#          finite-difference check -- which compares the implementation
#          against ITSELF -- cannot see it.
#   M17 the Cox gradient's risk-set sum is taken over the WRONG set
#       -> the gradient (C10).
#   M18 the Cox partial log-likelihood is the RAW SUM
#       -> the mean convention (C11).
#   M19 the Cox gradient's mean divides by n_events - 1
#       -> the normaliser (C11).
#   M20 the Cox Efron step uses the deaths' sum instead of the risk set
#       -> the Efron denominator (C9).
#   M21 the row-count guard is dropped
#       -> the guard that turns a shape mismatch into an ERROR rather than
#          a wrong number.
#   M22 the event-indicator guard is dropped
#       -> that guard.
#   M23 the "no events" guard is dropped
#       -> that guard. Without it the partial likelihood is empty and the
#          gradient is identically zero -- a silently useless answer.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY PAID FOR
# =============================================================================
#
#   1. `restore()` must NEVER delete its backup. It is called once PER
#      MUTATION and once on EXIT, so a self-cleaning restore works exactly
#      once and every later restore is a silent no-op -- leaving a MUTANT
#      IN THE TREE. Cleanup happens exactly once, on EXIT, and restore() is
#      ATOMIC (`cp tmp && mv -f`), because a plain `cp` truncates the
#      destination first, so a concurrent reader can splice two generations
#      together.
#   2. The link set comes from `make print-lib-objs`, NEVER from
#      regex-parsing the Makefile. `LIB_OBJS` is a RECURSIVE make variable
#      whose literal text contains no `.o` filenames, so a regex yields an
#      empty set and every mutation scores MUT-BROKEN -- one defect
#      reported as N verdicts. The count is asserted >= 10.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET. It legitimately
#      IS in the library's list (it is the TU being mutated), so the
#      correct handling is to compile the mutated TU SEPARATELY, drop the
#      library's copy from the link set, and link the mutant in its place.
#      Excluding it without replacing it breaks the link for the wrong
#      reason.
#   4. `MUT-BROKEN` means "the build failed". It is a separate class and is
#      never counted as a pass. A mutation reported BROKEN is re-run in
#      isolation before it is believed.
#   5. A CRASH (rc >= 128) is a THIRD class, not a verdict.
#   6. A mutation that DID NOT APPLY is a HARNESS DEFECT, not a
#      "SURVIVED" verdict. The tree is checksummed before and after every
#      edit.
#   7. The anchor must occur EXACTLY ONCE, and the count comes from the
#      SAME engine that performs the edit (`str.count` in Python), because
#      `grep -c` counts MATCHING LINES, not matches.
#   8. `rm -f` the intermediates before every build -- a stale .o from the
#      previous mutation reproduces a build failure that does not exist
#      alone.
#   9. NEVER run a build concurrently with this harness.
#  10. The tree is rebuilt and re-verified ON EXIT before any verdict is
#      reported.
#  11. The pristine tree must be GREEN before any mutation runs. A harness
#      that starts from a red tree reports garbage.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 99

SRC_CPP="include/nn/utils/survival.cpp"
SRC_HDR="include/nn/utils/survival.h"
MUT_OBJ="build/utils/survival.o"
SUITE="build/test_survival"

BAK_CPP="$(mktemp)"; BAK_HDR="$(mktemp)"
cp "$SRC_CPP" "$BAK_CPP"
cp "$SRC_HDR" "$BAK_HDR"

SCRATCH="$(mktemp -d)"

restore() {
    cp "$BAK_CPP" "$SRC_CPP.tmp" 2>/dev/null && mv -f "$SRC_CPP.tmp" "$SRC_CPP"
    cp "$BAK_HDR" "$SRC_HDR.tmp" 2>/dev/null && mv -f "$SRC_HDR.tmp" "$SRC_HDR"
}

cleanup() {
    restore
    rm -f "$BAK_CPP" "$BAK_HDR"
    rm -rf "$SCRATCH"
    # THE POST-HARNESS REBUILD-AND-VERIFY DEFENCE. A previous session shipped
    # a restored source with a stale mutant still on disk and reported a
    # phantom regression; this is the check that catches it.
    rm -f "$MUT_OBJ" build/test_survival.o "$SUITE"
    make setup >/dev/null 2>&1
    if make -j6 "$SUITE" >/dev/null 2>&1; then
        if "$SUITE" >/dev/null 2>&1; then
            echo ""
            echo "POST-HARNESS: tree restored, rebuilt, and GREEN."
        else
            echo ""
            echo "POST-HARNESS FAILURE: the restored tree does not pass. STOP." >&2
            exit 1
        fi
    else
        echo ""
        echo "POST-HARNESS FAILURE: the restored tree does not BUILD. STOP." >&2
        exit 1
    fi
}
trap cleanup EXIT INT TERM

checksum() { md5sum "$1" | cut -d' ' -f1; }

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
        *survival.o|*test_survival.o) ;;
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
# The test translation unit is compiled ONCE, up front: it depends on
# survival.h, which no mutation touches (every mutation is in the .cpp), so
# recompiling it per mutation would multiply the harness's runtime for no
# extra coverage. The mutated TU IS recompiled every time.
# -----------------------------------------------------------------------------
prepare_tests() {
    g++ $CXXFLAGS_LOCAL -c "tests/test_survival.cpp" -o "$SCRATCH/test_survival.o" 2>"$SCRATCH/build.log" || return 90
    return 0
}

build_suites() {
    rm -f "$MUT_OBJ" "$SCRATCH/survival.o" "$SCRATCH/test_survival"
    g++ $CXXFLAGS_LOCAL -c "$SRC_CPP" -o "$SCRATCH/survival.o" 2>"$SCRATCH/build.log" || return 90
    g++ $CXXFLAGS_LOCAL "$SCRATCH/test_survival.o" "$SCRATCH/survival.o" $LIB_LINK \
        -o "$SCRATCH/test_survival" 2>>"$SCRATCH/build.log" || return 90
    return 0
}

run_suite() {  # $1 = suite name
    local out rc
    out=$("$SCRATCH/$1" 2>&1); rc=$?
    LAST_OUT="$out"
    return $rc
}

failed_count() {
    echo "$1" | sed -nE 's/.*[Ss]ummary: ([0-9]+) passed, ([0-9]+) failed.*/\2/p' | tail -1
}

passed_count() {
    echo "$1" | sed -nE 's/.*[Ss]ummary: ([0-9]+) passed, ([0-9]+) failed.*/\1/p' | tail -1
}

# -----------------------------------------------------------------------------
# The pristine self-check: the suite must be GREEN before any mutation runs.
# -----------------------------------------------------------------------------
prepare_tests
if [ $? -ne 0 ]; then
    echo "HARNESS DEFECT: the test translation unit does not compile." >&2
    tail -20 "$SCRATCH/build.log" >&2
    exit 99
fi
build_suites
if [ $? -ne 0 ]; then
    echo "HARNESS DEFECT: the PRISTINE tree does not build." >&2
    tail -20 "$SCRATCH/build.log" >&2
    exit 99
fi
PRISTINE_OK=1
run_suite test_survival || { echo "HARNESS DEFECT: pristine suite exits non-zero." >&2; PRISTINE_OK=0; }
nf=$(failed_count "$LAST_OUT")
if [ -z "$nf" ] || [ "$nf" -ne 0 ]; then
    echo "HARNESS DEFECT: pristine suite is not green (${nf:-no summary} failed)." >&2
    PRISTINE_OK=0
else
    echo "pristine: test_survival -- $(passed_count "$LAST_OUT") passed, 0 failed"
fi
[ "$PRISTINE_OK" -eq 1 ] || exit 99
echo "pristine: green -- mutation may begin"

# -----------------------------------------------------------------------------
# Apply a mutation with a Python editor (str.count / str.replace). perl's
# s/// dies on C++ anchors containing '/', and `grep -c` counts LINES.
# The spec is "old\x1fnew" -- 0x1f cannot occur in either side.
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

# THE SEPARATOR. 0x1f (the "unit separator") cannot occur in either side of
# a mutation spec, so it is safe as a delimiter -- unlike '/', which C++ is
# FULL of and which killed a previous harness's perl s/// script.
#
# **IT MUST BE A REAL BYTE.** `$'\x1f'` produces it; `"\\x1f"` inside
# double quotes produces the four literal characters, and Python's
# `split("\x1f")` then finds no separator and raises ValueError -- which
# is what the FIRST DRAFT of a sibling harness did, and every mutation was
# reported HARNESS-DEFECT while the verdict table still looked plausible.
US=$'\x1f'

CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0
declare -a TABLE

print_table() {
    printf '%-6s %-54s %-42s %s\n' "ID" "MUTATION" "VERDICT" "DETAIL"
    printf '%-6s %-54s %-42s %s\n' "----" "------------------------------------------------------" "------------------------------------------" "------"
    local row a b c d
    for row in "${TABLE[@]}"; do
        IFS='|' read -r a b c d <<< "$row"
        printf '%-6s %-54s %-42s %s\n' "$a" "$b" "$c" "${d:-}"
    done
}

# -----------------------------------------------------------------------------
# run_mutation <id> <description> <old> <new>
# -----------------------------------------------------------------------------
run_mutation() {
    local id="$1" desc="$2" old="$3" new="$4"
    restore

    local out
    out=$(apply_mutation "$old${US}$new" "$SRC_CPP")
    if ! echo "$out" | grep -q "MUTATION_APPLIED"; then
        TABLE+=("$id|$desc|HARNESS-DEFECT (anchor count != 1)|$(echo "$out" | tr '\n' ' ')")
        NOOP=$((NOOP+1)); return
    fi

    # Verify the edit really landed.
    local before after
    before=$(checksum "$BAK_CPP"); after=$(checksum "$SRC_CPP")
    if [ "$before" = "$after" ]; then
        restore
        TABLE+=("$id|$desc|HARNESS-DEFECT (edit did not land)|")
        NOOP=$((NOOP+1)); return
    fi

    build_suites
    local rc=$?
    if [ "$rc" -eq 90 ]; then
        restore
        TABLE+=("$id|$desc|MUT-BROKEN (build failed -- NOT a verdict)|")
        BROKEN=$((BROKEN+1)); return
    fi

    run_suite test_survival; rc=$?
    local suite_out="$LAST_OUT"
    local nf
    nf=$(failed_count "$suite_out")
    if [ "$rc" -ge 128 ]; then
        restore
        TABLE+=("$id|$desc|MUT-CRASH (rc=$rc -- NOT a verdict)|")
        CRASH=$((CRASH+1)); return
    fi
    if [ -n "$nf" ] && [ "$nf" -gt 0 ]; then
        local line
        line=$(echo "$suite_out" | grep -m1 "FAIL" | sed 's/^ *//' | cut -c1-70)
        TABLE+=("$id|$desc|CAUGHT|$line")
        CAUGHT=$((CAUGHT+1))
    else
        TABLE+=("$id|$desc|SURVIVED (the suite stayed green)|")
        SURVIVED=$((SURVIVED+1))
    fi
    restore
}

echo
echo "=== survival mutations ==="

# --- M1: the risk-set convention -------------------------------------------
run_mutation M1 "the risk set is {t_j > t} instead of {t_j >= t}" \
    "            if (times(i, 0) >= t) {" \
    "            if (times(i, 0) > t) {"

# --- M2: the Kaplan-Meier product ------------------------------------------
run_mutation M2 "the KM factor uses n_risk - 1" \
    "        s *= (1.0 - g.events[k] / g.at_risk[k]);" \
    "        s *= (1.0 - g.events[k] / (g.at_risk[k] - 1.0));"

# --- M3: the right-continuity ----------------------------------------------
run_mutation M3 "the survival curve is LEFT-continuous" \
    "        if (grid[k] <= t) { idx = k; found = true; }" \
    "        if (grid[k] < t) { idx = k; found = true; }"

# --- M4: Greenwood's variance ----------------------------------------------
run_mutation M4 "the Greenwood term divides by n_risk only" \
    "            greenwood += g.events[k] / (nr * (nr - g.events[k]));" \
    "            greenwood += g.events[k] / nr;"

# --- M5: the n_risk == d limit ---------------------------------------------
run_mutation M5 "the Greenwood term is NOT guarded against n_risk == d" \
    "        if (nr - g.events[k] > 0.0)
            greenwood += g.events[k] / (nr * (nr - g.events[k]));" \
    "        greenwood += g.events[k] / (nr * (nr - g.events[k]));"

# --- M6: the median rule ---------------------------------------------------
run_mutation M6 "the median is the first time S is STRICTLY below 0.5" \
    "        if (survival[k] <= 0.5) return times[k];" \
    "        if (survival[k] < 0.5) return times[k];"

# --- M7: the Nelson-Aalen estimator ----------------------------------------
run_mutation M7 "the Nelson-Aalen hazard uses n_risk - d" \
    "        cum += g.events[k] / g.at_risk[k];" \
    "        cum += g.events[k] / (g.at_risk[k] - g.events[k]);"

# --- M8: the event grid includes the CENSORING times -----------------------
# **THE FIRST FORM OF THIS MUTATION CRASHED (rc=139), AND A CRASH IS NOT
# A VERDICT.** It replaced the Nelson-Aalen's grid with ALL the distinct
# times while leaving the hazard vector at the event-grid length, so the
# step lookup indexed past the end of a shorter vector. **A MUTATION THAT
# MAKES THE BINARY CRASH SAYS NOTHING ABOUT WHETHER THE SUITE COVERS THE
# BEHAVIOUR** -- it is a defect in the mutation, not evidence about the
# suite. The form below changes the grid at its SOURCE (the shared
# builder), so every vector stays the same length as its grid and the
# mutation changes the OBSERVABLE instead of the memory safety.
run_mutation M8 "the event grid includes the CENSORING times" \
    "    for (size_t i = 0; i < times.rows; ++i) {
        if (events(i, 0) == 1.0) {
            double t = times(i, 0);" \
    "    for (size_t i = 0; i < times.rows; ++i) {
        if (true) {
            double t = times(i, 0);"

# --- M9: the log-rank expectation ------------------------------------------
run_mutation M9 "the log-rank expected count divides by n-1" \
    "        res.expected_1 += d * n1 / n;" \
    "        res.expected_1 += d * n1 / (n - 1.0);"

# --- M10: the hypergeometric variance --------------------------------------
run_mutation M10 "the log-rank variance drops the finite-population factor" \
    "            res.variance += d * (n1 / n) * (1.0 - n1 / n) * ((n - d) / (n - 1.0));" \
    "            res.variance += d * (n1 / n) * (1.0 - n1 / n);"

# --- M11: the p-value's sign -----------------------------------------------
run_mutation M11 "the log-rank p-value is 1 - erfc(sqrt(stat/2))" \
    "    res.p_value = std::erfc(std::sqrt(res.statistic / 2.0));" \
    "    res.p_value = 1.0 - std::erfc(std::sqrt(res.statistic / 2.0));"

# --- M12: the statistic ----------------------------------------------------
run_mutation M12 "the log-rank statistic divides by V twice" \
    "        res.statistic = (res.observed_1 - res.expected_1) * (res.observed_1 - res.expected_1) /
                        res.variance;" \
    "        res.statistic = (res.observed_1 - res.expected_1) * (res.observed_1 - res.expected_1) /
                        (res.variance * res.variance);"

# --- M13: the concordance tie credit ---------------------------------------
run_mutation M13 "the concordance drops the half credit for tied scores" \
    "                res.num_tied += 1.0;" \
    "                res.num_tied += 1.0;
                res.num_correct += 1.0;"

# --- M14: THE CONCORDANCE SIGN ---------------------------------------------
run_mutation M14 "the concordance counts the LATER failure as correct" \
    "                if (risk_first > risk_second) res.num_correct += 1.0;" \
    "                if (risk_first < risk_second) res.num_correct += 1.0;"

# --- M15: the admissibility rule -------------------------------------------
run_mutation M15 "the concordance never admits an equal-times pair" \
    "                if (ea == eb) continue;" \
    "                if (true) continue;"

# --- M16: THE COX BRESLOW DENOMINATOR --------------------------------------
# **THIS MUTATION IS THE REASON THE SUITE SHIPS AN INDEPENDENT
# SCALAR-LOOP REFERENCE.** It moves the value AND the gradient
# CONSISTENTLY, so a finite-difference check -- which compares the
# implementation against ITSELF -- cannot see it.
run_mutation M16 "the Cox Breslow denominator keeps the tied deaths'" \
    "            loglik += sum_eta_deaths - dt * std::log(s0);" \
    "            loglik += sum_eta_deaths - std::log(s0);"

# --- M17: the Cox gradient's risk-set sum ----------------------------------
run_mutation M17 "the Cox gradient's risk-set sum is over the deaths only" \
    "                s0 += w;
                for (size_t j = 0; j < D; ++j) s0_x[j] += w * X(i, j);" \
    "                s0 += w;
                if (!is_death) for (size_t j = 0; j < D; ++j) s0_x[j] += w * X(i, j);"

# --- M18: the mean convention ----------------------------------------------
run_mutation M18 "the Cox partial log-likelihood is the RAW SUM" \
    "    res.log_likelihood = loglik / n_events;" \
    "    res.log_likelihood = loglik;"

# --- M19: the gradient's normaliser ----------------------------------------
run_mutation M19 "the Cox gradient's mean divides by n_events - 1" \
    "    for (size_t j = 0; j < D; ++j) grad(0, j) /= n_events;" \
    "    for (size_t j = 0; j < D; ++j) grad(0, j) /= (n_events - 1.0);"

# --- M20: the Efron denominator --------------------------------------------
run_mutation M20 "the Cox Efron step uses the deaths' sum, not the risk set" \
    "                double denom = s0 - frac * s0_d;" \
    "                double denom = s0_d - frac * s0_d;"

# --- M21: the row-count guard ----------------------------------------------
run_mutation M21 "the row-count guard is dropped" \
    "    if (times.rows != events.rows)
        throw std::invalid_argument(std::string(who) +
                                    \": the times and the events have different \"
                                    \"row counts\");" \
    "    if (false)
        throw std::invalid_argument(std::string(who) +
                                    \": the times and the events have different \"
                                    \"row counts\");"

# --- M22: the event-indicator guard ----------------------------------------
run_mutation M22 "the event-indicator guard is dropped" \
    "        if (e != 0.0 && e != 1.0)" \
    "        if (false)"

# --- M23: the no-events guard ----------------------------------------------
run_mutation M23 "the Cox 'no events' guard is dropped" \
    "    if (n_events == 0.0)
        throw std::invalid_argument(std::string(\"cox_partial_log_likelihood\") +
                                    \": the dataset contains no events, so the \"
                                    \"partial likelihood is empty\");" \
    "    if (false)
        throw std::invalid_argument(std::string(\"cox_partial_log_likelihood\") +
                                    \": the dataset contains no events, so the \"
                                    \"partial likelihood is empty\");"

echo
echo "=== caught $CAUGHT / survived $SURVIVED / broken $BROKEN / crashed $CRASH / noop $NOOP ==="
echo ""
print_table
echo ""
echo "FINAL: $CAUGHT caught / $SURVIVED survived / $BROKEN broken / $CRASH crashed / $NOOP noop"

[ "$SURVIVED" -eq 0 ] || { echo "NOTE: $SURVIVED unexplained survivor(s) -- these are REAL coverage gaps." >&2; exit 3; }
[ "$BROKEN" -eq 0 ] || { echo "NOTE: $BROKEN mutation(s) reported BROKEN/CRASH -- re-run in isolation." >&2; exit 1; }
[ "$CRASH" -eq 0 ] || { echo "NOTE: $CRASH mutation(s) reported CRASH -- re-run in isolation." >&2; exit 1; }
[ "$NOOP" -eq 0 ] || { echo "NOTE: $NOOP harness defect(s)." >&2; exit 1; }
exit 0
