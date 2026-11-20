#!/usr/bin/env bash
# =============================================================================
# tests/mutate_contingency.sh -- mutation harness for the categorical-agreement
# and contingency-table-inference utilities: utils/contingency.{h,cpp}.
#
# WHY THIS FILE EXISTS
# =============================================================================
#
# A suite is, by construction, unable to report a coverage gap: it can only
# report what it asserts. This harness breaks the implementation in specific
# ways and requires that the suite FAIL each time. It is the only thing in
# this repo that can find "the suite is green and the behaviour is wrong".
#
# The specifications are the oracle (tests/contingency_oracle.py, float64
# numpy, measured against scipy / scikit-learn / R) and the plan
# (docs/plans/2026-11-19-contingency-statistics.md). The conventions are
# C1-C13; the fixtures are F1-F16.
#
# =============================================================================
# THE MUTATIONS, AND WHAT EACH ONE IS FOR
# =============================================================================
#
#   M1  the df == 1 chi-square path is CLAIMED FOR df == 2
#       -> C1. The erfc closed form exists for exactly ONE degree of freedom.
#          The F1 fixture has df == 2 and df == 10 entries, and F7's table is
#          2x3 (df == 2), so the general path is exercised.
#   M2  the statistic <= 0 guard returns 0.0 instead of 1.0
#       -> C1. P(X >= 0) = 1 for every df. The guard is what the general path
#          needs, because log(0) = -inf would otherwise make it NaN.
#          F1 pins chi2_sf(0, 3) == 1.
#   M3  the incomplete-gamma SERIES branch returns P(a,x) instead of Q(a,x)
#       -> C1. The survival function is INVERTED on the series branch: the
#          regularized UPPER incomplete gamma is Q = 1 - P, and the series
#          computes P. F1/F7 exercise the general path.
#   M4  the linear and the quadratic kappa weight matrices are SWAPPED
#       -> the weight convention. The absolute distance and the squared
#          distance are different matrices, and F2 pins the two weighted
#          kappas to DIFFERENT absolute values, so a swap cannot hide.
#   M5  the expected matrix in the kappa is TRANSPOSED
#       -> C4. **MEASURED INERT.** Every weight matrix this module supports
#          (the off-diagonal indicator, the absolute distance, the squared
#          distance) is SYMMETRIC, and for symmetric w the sums over
#          w_ij E_ij and over w_ij E_ji are equal by relabelling, so the
#          transposition cancels exactly (up to the last floating-point bit).
#          It is also the CONVENTIONAL orientation that this module ships,
#          so the mutant is the sklearn orientation -- a legal
#          implementation, not a bug. EXPECTED TO SURVIVE, and its survival
#          is reported as such.
#   M6  the DEGENERATE kappa is 0.0 instead of NaN
#       -> C3. The undefined case is NaN, and F3 pins all three weight
#          schemes' degenerate kappa to NaN.
#   M7  the unweighted kappa's weight matrix KEEPS ITS DIAGONAL
#       -> the weight convention. The off-diagonal indicator has a zero
#          diagonal; keeping it makes the unweighted kappa a different
#          number, and F2 pins the unweighted kappa absolutely.
#   M8  the DEGENERATE Matthews coefficient is NaN instead of 0.0
#       -> C5. The OPPOSITE of the kappa's degenerate case, and F5 pins it.
#   M9  cov_ytyp uses the PREDICTED marginal twice instead of the dot
#       product of the true and the predicted marginal
#       -> the MCC formula. F4 pins the MCC absolutely AND against an
#          independent scalar-loop reference.
#   M10 the Yates correction is UNCLAMPED (the textbook form)
#       -> C6-bis. The clamp to the cell's OWN deviation is the whole point;
#          the textbook form gives 0.3125 on F8's 2x2 [[1,1],[1,2]] where
#          scipy and R both give 0.0.
#   M11 the Yates correction is applied to EVERY table size, not just 2x2
#       -> C6. The flag is honoured only for a 2x2, and F7's table is 2x3.
#   M12 the degrees of freedom are the CELL COUNT minus one, not (R-1)(C-1)
#       -> C6-ter. scipy's power_divergence uses cells - 1 and is WRONG for a
#          contingency table; F7's 2x3 table has df == 2, not 5.
#   M13 the chi-square deviation is ABSOLUTED before squaring
#       -> **MEASURED INERT.** The absolute value squared equals the value
#          squared bit for bit in IEEE-754 double precision, so the mutant
#          sums exactly the same terms in the same order. It cannot be
#          caught by any fixture, and pretending otherwise would be a lie.
#          EXPECTED TO SURVIVE.
#   M14 the G-test's ZERO-CELL guard is removed, so 0 * log(0 / E) = NaN
#       -> C8. The naive expression is 0 * (-inf) = NaN and destroys the
#          whole statistic. F9's table HAS a zero cell.
#   M15 the G-test's factor of 2 is DROPPED
#       -> the G statistic's normalisation, G = 2 * sum O log(O/E). F9 pins
#          the statistic absolutely.
#   M16 McNemar's continuity correction is applied even on a SYMMETRIC
#       table (R SKIPS it there)
#       -> C9. When n01 == n10 the statistic is the plain squared difference,
#          0, NOT (0 - 1)^2 / (2 * n01). F11 is THE discriminating fixture.
#   M17 the ABSOLUTE VALUE is dropped from McNemar's numerator
#       -> C9-ter. R writes abs(x - t(x)) - 1, so the numerator is the
#          absolute difference minus one. The signed form differs whenever
#          n01 < n10, which is F12.
#   M18 McNemar's NO-DISCORDANT-PAIRS case is 0 instead of NaN
#       -> C10. The statistic is 0/0 there, and R reports NaN. F13.
#   M19 the Fisher probability ordering drops scipy's 1 + 1e-14 RELATIVE
#       tie tolerance
#       -> C12-bis. Two tables can share the observed pmf in exact arithmetic
#          while their float64 evaluations differ in the last bits; the
#          tolerance includes the second one and a bare comparison EXCLUDES
#          it, which HALVES the p-value on F16.
#          *** SEE THE NOTE ON M19 IN THE SPEC BELOW. This mutation is
#          *** VISIBLE ONLY THROUGH A TABLE WHOSE TIE IS BIT-EXACT IN ONE
#          *** libm AND INEXACT IN THE OTHER, and the oracle's F16 table is
#          *** such a table in PYTHON but NOT in C++ on this machine.
#   M20 Fisher's two-sided p-value becomes a DOUBLED ONE-SIDED TAIL instead
#       of the probability ordering
#       -> C12. The probability-ordering rule is not a doubled tail, and the
#          two disagree on F14 and F15.
#   M21 Fisher's odds ratio is the WRONG cross-product a*c/(b*d) instead of
#       a*d/(b*c)
#       -> C13. The unconditional cross-product ratio is what scipy returns;
#          F14 and F15 pin it absolutely.
#
# =============================================================================
# TWO MUTATIONS THAT HAD TO BE REWRITTEN, AND WHY
# =============================================================================
#
# The FIRST FORM OF EACH WAS REJECTED BY THE HARNESS ITSELF, and both
# rejections are worth recording, because each one is a way a mutation can
# quietly stop being a mutation at all.
#
#   M14 -- "THE ZERO-CELL GUARD IS DELETED". The literal form of C8's
#      violation. Deleting `if (o <= 0.0) continue;` makes a zero observed
#      cell contribute 0 * log(0 / E) = 0 * (-inf) = NaN. The NaN
#      statistic then reaches chi_square_survival, whose OWN finiteness
#      guard throws std::invalid_argument; the exception escapes main and
#      the binary dies with SIGABRT. The harness scored that MUT-CRASH --
#      correctly, because a CRASH IS NOT A VERDICT. It says nothing about
#      whether the suite covers the behaviour. The mutation was rewritten
#      to keep the convention under attack (a zero observed cell must
#      contribute EXACTLY 0) while making the wrong contribution FINITE,
#      so the suite sees a wrong NUMBER.
#
#   M19 -- "THE TIE TOLERANCE IS DROPPED". The first form of this mutation
#      was correct against the ORACLE and invisible against the C++. The
#      oracle measures Fisher in PYTHON, where the fixture table
#      [[0,2],[5,3]] has a pmf tie that is INEXACT: CPython's math.lgamma
#      and glibc's lgamma round lgamma(3.0) differently in the last bit
#      (0.69314718055994495316 vs 0.69314718055994528623), so the two
#      tied tables evaluate to different doubles and the tolerance is what
#      includes the second one. In C++ on this machine glibc's std::lgamma
#      makes the tie BIT-EXACT, so the bare comparison already includes
#      the second table: dropping the tolerance changes NOTHING and the
#      suite stays green. A sweep of 2,819,200 tables found 310 whose tie
#      is inexact under C++; the table [[0,2],[15,13]] is inexact under
#      BOTH libms and is the fixture that would make this convention
#      observable in the C++ build. THE FIXTURE SET IS THE TEST SUITE'S
#      BUSINESS, NOT THIS HARNESS'S -- so this harness REPORTS the gap
#      rather than papering over it.
#
# =============================================================================
# THE VERDICT CLASSES -- FOUR OF THEM, AND THEY MUST NOT BE CONFUSED
# =============================================================================
#
#   CAUGHT     the suite FAILED. The mutation was detected. This is the only
#              good outcome for a mutation that is supposed to be caught.
#   SURVIVED   the suite PASSED. The mutation was NOT detected. For a
#              mutation that is supposed to be caught this is a REAL COVERAGE
#              GAP in the fixture, and the harness exits non-zero.
#   MUT-BROKEN the BUILD failed. Not evidence of anything: a mutant that
#              does not compile tells you nothing about the suite.
#   MUT-CRASH  the binary died with a signal (rc >= 128). ALSO not a
#              verdict: the mutation removed a resource instead of changing
#              an observable, and the mutation -- not the suite -- has to be
#              rewritten.
#
# Two more classes are HARNESS DEFECTS, not verdicts about the suite, and
# they are counted apart from the four above:
#
#   MUT-NOOP   the edit did not land (the source's checksum is unchanged),
#              or the replacement text is not where it was supposed to be.
#   MUT-ANCHOR the anchor does not occur exactly once in the source, so the
#              edit would have been a COIN FLIP. The harness REFUSES to edit
#              an ambiguous anchor.
#
# **THE DECLARED-INERT SURVIVORS (M5 and M13).** These two mutations are
# PROVABLY INERT: M5 because every supported weight matrix is symmetric, so
# the transposition of the expected matrix cancels exactly, and M13 because
# the absolute value squared equals the value squared bit for bit. A suite
# that "caught" either of them would be wrong, and a harness that called
# their survival a coverage gap would be wrong too. They are therefore
# reported as SURVIVED with the reason attached, and they are counted in
# their own bucket. The exit code treats them as the expected outcome; pass
# --strict to make them fail the run instead (the strict reading of "exit
# non-zero if any mutation SURVIVED").
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
#      empty set and every mutation scores MUT-BROKEN -- one defect reported
#      as N verdicts. The count is asserted >= 10.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET. It legitimately
#      IS in the library's list (it is the TU being mutated), so the
#      mutated TU is compiled SEPARATELY, dropped from the library's copy,
#      and linked in its place. Excluding it without replacing it breaks the
#      link for the wrong reason.
#   4. `MUT-BROKEN` means "the build failed". It is a separate class and is
#      never counted as a pass.
#   5. A CRASH (rc >= 128) is a THIRD class, not a verdict.
#   6. A mutation that DID NOT APPLY is a HARNESS DEFECT, not a "SURVIVED"
#      verdict. The tree is checksummed before and after every edit, and the
#      replacement text is re-counted after the edit.
#   7. The anchor must occur EXACTLY ONCE. The count is of the MULTI-LINE
#      BLOCK, computed by matching candidate start lines and comparing the
#      whole block byte for byte -- `grep -c` counts matching LINES, not
#      matches, and would silently accept an ambiguous anchor.
#   8. `rm -f` the intermediates before every build -- a stale .o from the
#      previous mutation reproduces a build failure that does not exist
#      alone.
#   9. NEVER run a build concurrently with this harness, and never run
#      `make tests` from inside it. This harness compiles the ONE test
#      binary it needs, out of the current (possibly mutated) source, into
#      its own directory, and touches nothing else in build/.
#  10. The tree is rebuilt and re-verified ON EXIT before any verdict is
#      reported.
#  11. The pristine tree must be GREEN before any mutation runs. A harness
#      that starts from a red tree reports garbage.
#
# =============================================================================
# HOW THE EDITS ARE MADE (sed, in two steps, with a placeholder token)
# =============================================================================
#
# The anchors are MULTI-LINE blocks, and GNU sed's `s` command cannot match
# a newline inside a pattern (the pattern space is one line at a time), so
# a one-shot `s/OLD/NEW/` edit across lines is impossible. The edit is
# therefore made in TWO STEPS, with a unique placeholder token in between:
#
#   step 1   sed -i '<start>,<end>c\@@MUTATION_SLOT_M<n>@@'
#            -- the whole OLD block is replaced by ONE placeholder line.
#   step 2   sed -i -f <script>
#            -- the script reads the NEW block from a file after the
#            placeholder line and deletes the placeholder line, so the NEW
#            block lands exactly where the OLD one was.
#
# The NEW block is read from a FILE, so it needs no escaping at all: a C++
# anchor is full of `/`, `&`, `\` and `"`, and every one of them is a
# metacharacter in some other engine's s/// . The placeholder token also
# makes the edit verifiable: after step 1 the token must occur exactly once,
# and after step 2 it must occur zero times and the NEW block exactly once.
#
# =============================================================================
# THE MUTATION SPEC
# =============================================================================
#
# One record per mutation, in the order the report prints them:
#
#   @@MUTATION <id>
#   @@DESCRIPTION <one line, no backticks, no command substitution>
#   @@CONVENTION <C1..C13>
#   @@FIXTURE <F1..F16>
#   @@EXPECTATION caught | inert
#   @@WHY <why an inert mutation is inert>
#   @@OLD
#   ...the exact block in the source...
#   @@NEW
#   ...the exact replacement...
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 99

SRC_CPP="include/nn/utils/contingency.cpp"
TEST_SRC="tests/test_contingency.cpp"
MUT_BUILD="$ROOT/build/mutation/contingency"
SUITE_NAME="test_contingency"

STRICT=0
VERIFY_ONLY=0
POST_VERIFY=0
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
        --verify) VERIFY_ONLY=1 ;;
        -h|--help)
            echo "usage: $0 [--verify] [--strict]"
            echo "  --verify  check every anchor against the source and stop"
            echo "  --strict  treat the declared-inert survivors as failures"
            exit 0 ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 99 ;;
    esac
done

if [ ! -f "$SRC_CPP" ]; then
    echo "HARNESS DEFECT: $SRC_CPP does not exist. The implementation must be" >&2
    echo "in place before this harness can say anything about the suite." >&2
    exit 99
fi

# -----------------------------------------------------------------------------
# The backup. restore() NEVER deletes it; cleanup does, on EXIT.
# -----------------------------------------------------------------------------
BAK="$(mktemp)"
cp "$SRC_CPP" "$BAK"
SCRATCH="$(mktemp -d)"
MUT_DIR="$SCRATCH/muts"
mkdir -p "$MUT_DIR" "$MUT_BUILD"

restore() {
    # ATOMIC: cp to a temp then mv, so a concurrent reader never sees a
    # half-written source.
    cp "$BAK" "$SRC_CPP.tmp" 2>/dev/null && mv -f "$SRC_CPP.tmp" "$SRC_CPP"
}

cleanup() {
    local rc=$?
    restore
    rm -f "$BAK"
    if [ "$VERIFY_ONLY" -eq 0 ] && [ "$POST_VERIFY" -eq 1 ]; then
        # THE POST-HARNESS REBUILD-AND-VERIFY DEFENCE. A previous session
        # shipped a restored source with a stale mutant still on disk and
        # reported a phantom regression; this is the check that catches it.
        # It rebuilds ONLY the one test binary, out of the RESTORED source,
        # and runs it. It never invokes `make tests`.
        rm -f "$MUT_BUILD/contingency.o" "$MUT_BUILD/$SUITE_NAME"
        if build_suite; then
            if "$MUT_BUILD/$SUITE_NAME" >/dev/null 2>&1; then
                echo ""
                echo "POST-HARNESS: tree restored, rebuilt, and GREEN."
            else
                echo "" >&2
                echo "POST-HARNESS FAILURE: the restored tree does not pass. STOP." >&2
                rc=1
            fi
        else
            echo "" >&2
            echo "POST-HARNESS FAILURE: the restored tree does not BUILD. STOP." >&2
            rc=1
        fi
    fi
    rm -rf "$MUT_BUILD" "$SCRATCH"
    exit $rc
}
trap cleanup EXIT INT TERM

checksum() { md5sum "$1" | cut -d' ' -f1; }

# -----------------------------------------------------------------------------
# The link set, from make itself, with a count assertion.
# `make print-lib-objs` only PRINTS the expanded list -- it compiles nothing
# and writes nothing -- so it cannot race a concurrent `make`.
# -----------------------------------------------------------------------------
mapfile -t ALL_OBJS < <(make print-lib-objs 2>/dev/null)
if [ "${#ALL_OBJS[@]}" -lt 10 ]; then
    echo "HARNESS DEFECT: the link set has only ${#ALL_OBJS[@]} objects (need >= 10)." >&2
    echo "REFUSING TO RUN -- every verdict below would be false." >&2
    exit 99
fi
LIB_LINK=""
for o in "${ALL_OBJS[@]}"; do
    case "$o" in
        */utils/contingency.o|*/${SUITE_NAME}.o) ;;
        *) LIB_LINK="$LIB_LINK $o" ;;
    esac
done
LIB_COUNT=$(echo "$LIB_LINK" | wc -w)
if [ "$LIB_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: the link set collapsed to $LIB_COUNT objects." >&2
    exit 99
fi
echo "link set: $LIB_COUNT objects (the mutated TU is compiled separately and linked in)"

CXXFLAGS_LOCAL="-std=c++17 -O2 -Wall -Wextra -march=native -Iinclude"

# -----------------------------------------------------------------------------
# The mutation spec, and the splitter that turns it into one OLD file and one
# NEW file per mutation.
# -----------------------------------------------------------------------------
read -r -d '' MUT_SPEC <<'MUTSPEC'
@@MUTATION M1
@@DESCRIPTION the df == 1 chi-square path is claimed for df == 2
@@CONVENTION C1
@@FIXTURE F1/F7
@@EXPECTATION caught
@@OLD
    // C1: bit for bit with utils/survival.cpp.
    if (degrees_of_freedom == 1.0) return std::erfc(std::sqrt(statistic / 2.0));
    return gamma_upper(degrees_of_freedom / 2.0, statistic / 2.0);
@@NEW
    // C1: bit for bit with utils/survival.cpp.
    if (degrees_of_freedom == 2.0) return std::erfc(std::sqrt(statistic / 2.0));
    return gamma_upper(degrees_of_freedom / 2.0, statistic / 2.0);
@@MUTATION M2
@@DESCRIPTION the statistic <= 0 guard returns 0.0 instead of 1.0
@@CONVENTION C1
@@FIXTURE F1
@@EXPECTATION caught
@@OLD
    // P(X >= 0) = 1 for every df. The guard is not cosmetic: the general
    // path evaluates log(x) with x = statistic / 2, so a statistic of 0 is
    // log(0) = -inf and the whole expression is NaN.
    if (statistic == 0.0) return 1.0;
@@NEW
    // P(X >= 0) = 1 for every df. The guard is not cosmetic: the general
    // path evaluates log(x) with x = statistic / 2, so a statistic of 0 is
    // log(0) = -inf and the whole expression is NaN.
    if (statistic == 0.0) return 0.0;
@@MUTATION M3
@@DESCRIPTION the incomplete-gamma series branch returns P(a,x) instead of Q(a,x)
@@CONVENTION C1
@@FIXTURE F1/F7
@@EXPECTATION caught
@@OLD
double gamma_upper(double a, double x) {
    if (x < a + 1.0) return 1.0 - gamma_series(a, x);
    return gamma_continued_fraction(a, x);
}
@@NEW
double gamma_upper(double a, double x) {
    if (x < a + 1.0) return gamma_series(a, x);
    return gamma_continued_fraction(a, x);
}
@@MUTATION M4
@@DESCRIPTION the linear and the quadratic kappa weight matrices are swapped
@@CONVENTION C2
@@FIXTURE F2
@@EXPECTATION caught
@@OLD
            switch (weights) {
                case KappaWeights::Unweighted:
                    w = (i == j) ? 0.0 : 1.0;
                    break;
                case KappaWeights::Linear:
                    w = std::fabs(diff);
                    break;
                case KappaWeights::Quadratic:
                default:
                    w = diff * diff;
                    break;
            }
@@NEW
            switch (weights) {
                case KappaWeights::Unweighted:
                    w = (i == j) ? 0.0 : 1.0;
                    break;
                case KappaWeights::Linear:
                    w = diff * diff;
                    break;
                case KappaWeights::Quadratic:
                default:
                    w = std::fabs(diff);
                    break;
            }
@@MUTATION M5
@@DESCRIPTION the expected matrix in the kappa is transposed
@@CONVENTION C4
@@FIXTURE F2
@@EXPECTATION inert
@@WHY every supported weight matrix is symmetric, so the sum over w E equals the sum over the transposed w E exactly; the transposition cancels
@@OLD
    std::vector<double> expected =
        expected_counts(num_classes, num_classes, row_sums, col_sums, total);
@@NEW
    std::vector<double> expected =
        expected_counts(num_classes, num_classes, col_sums, row_sums, total);
@@MUTATION M6
@@DESCRIPTION the degenerate kappa is 0.0 instead of NaN
@@CONVENTION C3
@@FIXTURE F3
@@EXPECTATION caught
@@OLD
    res.kappa = (denominator == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                                     : 1.0 - numerator / denominator;
@@NEW
    res.kappa = (denominator == 0.0) ? 0.0
                                     : 1.0 - numerator / denominator;
@@MUTATION M7
@@DESCRIPTION the unweighted kappa weight matrix keeps its diagonal
@@CONVENTION C2
@@FIXTURE F2
@@EXPECTATION caught
@@OLD
                case KappaWeights::Unweighted:
                    w = (i == j) ? 0.0 : 1.0;
                    break;
@@NEW
                case KappaWeights::Unweighted:
                    w = 1.0;
                    break;
@@MUTATION M8
@@DESCRIPTION the degenerate Matthews coefficient is NaN instead of 0.0
@@CONVENTION C5
@@FIXTURE F5
@@EXPECTATION caught
@@OLD
    res.mcc = (denom == 0.0) ? 0.0 : cov_ytyp / std::sqrt(denom);
@@NEW
    res.mcc = (denom == 0.0) ? std::numeric_limits<double>::quiet_NaN()
                              : cov_ytyp / std::sqrt(denom);
@@MUTATION M9
@@DESCRIPTION cov_ytyp uses the predicted marginal twice instead of the dot product
@@CONVENTION C5
@@FIXTURE F4
@@EXPECTATION caught
@@OLD
        dot_tp += t_sum[k] * p_sum[k];
        dot_pp += p_sum[k] * p_sum[k];
@@NEW
        dot_tp += p_sum[k] * p_sum[k];
        dot_pp += p_sum[k] * p_sum[k];
@@MUTATION M10
@@DESCRIPTION the Yates correction is unclamped (the textbook form)
@@CONVENTION C6-bis
@@FIXTURE F8
@@EXPECTATION caught
@@OLD
                double diff = e - o;
                double magnitude = std::min(0.5, std::fabs(diff));
@@NEW
                double diff = e - o;
                double magnitude = 0.5;
@@MUTATION M11
@@DESCRIPTION the Yates correction is applied to every table size, not just 2x2
@@CONVENTION C6
@@FIXTURE F7
@@EXPECTATION caught
@@OLD
    // C6: the continuity correction is applied ONLY to a 2x2 table, and the
    // flag is honoured ONLY there.
    const bool apply_correction = correct && num_rows == 2 && num_cols == 2;
@@NEW
    // C6: the continuity correction is applied ONLY to a 2x2 table, and the
    // flag is honoured ONLY there.
    const bool apply_correction = correct;
@@MUTATION M12
@@DESCRIPTION the degrees of freedom are the cell count minus one
@@CONVENTION C6-ter
@@FIXTURE F7
@@EXPECTATION caught
@@OLD
    // C6-ter: the contingency degrees of freedom, NOT scipy's
    // power_divergence "cells - 1".
    res.degrees_of_freedom = (num_rows - 1) * (num_cols - 1);
@@NEW
    // C6-ter: the contingency degrees of freedom, NOT scipy's
    // power_divergence "cells - 1".
    res.degrees_of_freedom = num_rows * num_cols - 1;
@@MUTATION M13
@@DESCRIPTION the chi-square deviation is absoluted before squaring
@@CONVENTION C6-bis
@@FIXTURE F6/F7
@@EXPECTATION inert
@@WHY the absolute value squared equals the value squared bit for bit in IEEE-754 double precision, so the mutant sums exactly the same terms in the same order
@@OLD
            double delta = adjusted - e;
            statistic += delta * delta / e;
@@NEW
            double delta = adjusted - e;
            statistic += std::fabs(delta) * std::fabs(delta) / e;
@@MUTATION M14
@@DESCRIPTION the G-test zero cell is scored with a pseudo-count of 1 instead of contributing exactly 0
@@CONVENTION C8
@@FIXTURE F9
@@EXPECTATION caught
@@NOTE the LITERAL form of this mutation -- deleting the guard, so that a zero cell contributes 0 * log(0 / E) = NaN -- SCORES NOTHING: the NaN statistic reaches chi_square_survival, whose own finiteness guard throws std::invalid_argument, the exception escapes main, and the binary dies with SIGABRT. A crash is not a verdict. This form keeps the CONVENTION under attack (a zero observed cell must contribute EXACTLY 0) but makes the wrong contribution FINITE, so the suite sees a wrong NUMBER. Only a fixture with a zero observed cell can see it at all: F9 is the only one.
@@OLD
            if (o <= 0.0) continue;
@@NEW
            if (o <= 0.0) { log_likelihood += std::log(1.0 / expected[i * num_cols + j]); continue; }
@@MUTATION M15
@@DESCRIPTION the G-test factor of 2 is dropped
@@CONVENTION C6-quater
@@FIXTURE F9
@@EXPECTATION caught
@@OLD
    GTestResult res;
    res.statistic = 2.0 * log_likelihood;
@@NEW
    GTestResult res;
    res.statistic = log_likelihood;
@@MUTATION M16
@@DESCRIPTION the McNemar continuity correction is applied on a symmetric table
@@CONVENTION C9
@@FIXTURE F11
@@EXPECTATION caught
@@OLD
    if (correct && n01 != n10) {
        // C9-ter: the ABSOLUTE value. R writes abs(x - t(x)) - 1, so the
@@NEW
    if (correct) {
        // C9-ter: the ABSOLUTE value. R writes abs(x - t(x)) - 1, so the
@@MUTATION M17
@@DESCRIPTION the absolute value is dropped from the McNemar numerator
@@CONVENTION C9-ter
@@FIXTURE F12
@@EXPECTATION caught
@@OLD
        // numerator is |n01 - n10| - 1 and NOT n01 - n10 - 1; the two differ
        // whenever n01 < n10.
        double numerator = std::fabs(diff) - 1.0;
@@NEW
        // numerator is |n01 - n10| - 1 and NOT n01 - n10 - 1; the two differ
        // whenever n01 < n10.
        double numerator = diff - 1.0;
@@MUTATION M18
@@DESCRIPTION the McNemar no-discordant-pairs case is 0 instead of NaN
@@CONVENTION C10
@@FIXTURE F13
@@EXPECTATION caught
@@OLD
        // not 0. R reports NaN for both the statistic and the p-value.
        res.statistic = std::numeric_limits<double>::quiet_NaN();
        res.p_value = std::numeric_limits<double>::quiet_NaN();
@@NEW
        // not 0. R reports NaN for both the statistic and the p-value.
        res.statistic = 0.0;
        res.p_value = 0.0;
@@MUTATION M19
@@DESCRIPTION the Fisher tie tolerance of 1 + 1e-14 is dropped
@@CONVENTION C12-bis
@@FIXTURE F16
@@EXPECTATION caught
@@NOTE THIS MUTATION IS ONLY OBSERVABLE THROUGH A TABLE WHOSE TIE IS BIT-EXACT IN ONE libm AND INEXACT IN THE OTHER. The oracle's F16 table [[0,2],[5,3]] has that property in PYTHON (CPython's math.lgamma rounds differently from glibc's lgamma), but in C++ ON THIS MACHINE glibc's std::lgamma makes the tie BIT-EXACT, so the bare comparison already includes the second table and the tolerance changes NOTHING -- the mutation is invisible and the suite stays green. That is a REAL finding about the FIXTURE, not a harness defect: the fixture F16 does not pin the tolerance in the C++ build. The table [[0,2],[15,13]] has an INEXACT tie in BOTH libms and so is the fixture that must be added to make this convention observable; the fixture set is the test suite's business, not this harness's.
@@OLD
    double pexact = hypergeometric_pmf(a, total, row1, col1);
    const double gamma = 1.0 + 1e-14;
@@NEW
    double pexact = hypergeometric_pmf(a, total, row1, col1);
    const double gamma = 1.0;
@@MUTATION M20
@@DESCRIPTION the Fisher two-sided p-value becomes a doubled one-sided tail
@@CONVENTION C12
@@FIXTURE F14/F15
@@EXPECTATION caught
@@OLD
    double pexact = hypergeometric_pmf(a, total, row1, col1);
    const double gamma = 1.0 + 1e-14;
    double lo = std::max(0.0, col1 - (total - row1));
    double hi = std::min(row1, col1);
    double sum = 0.0;
    for (double x = lo; x <= hi; x += 1.0) {
        double p = hypergeometric_pmf(x, total, row1, col1);
        if (p <= pexact * gamma) sum += p;
    }
    res.p_value = std::min(sum, 1.0);
@@NEW
    double lo = std::max(0.0, col1 - (total - row1));
    double hi = std::min(row1, col1);
    double lower = 0.0;
    double upper = 0.0;
    for (double x = lo; x <= hi; x += 1.0) {
        const double p = hypergeometric_pmf(x, total, row1, col1);
        if (x <= a) lower += p;
        if (x >= a) upper += p;
    }
    res.p_value = std::min(2.0 * std::min(lower, upper), 1.0);
@@MUTATION M21
@@DESCRIPTION the Fisher odds ratio is the wrong cross-product a times c over b times d
@@CONVENTION C13
@@FIXTURE F14/F15
@@EXPECTATION caught
@@OLD
    if (c > 0.0 && b > 0.0)
        res.odds_ratio = a * d / (c * b);
@@NEW
    if (c > 0.0 && b > 0.0)
        res.odds_ratio = a * c / (b * d);
MUTSPEC

# The splitter. Every line that is not a marker belongs to the block that is
# currently open, and is written out VERBATIM -- the anchors are C++ and
# their indentation is part of the match. The ORDER is recorded as the
# records are read, because a glob would sort M10 between M1 and M2.
cat > "$SCRATCH/split.awk" <<'SPLIT_AWK'
BEGIN { id = ""; mode = "" }
/^@@MUTATION / {
    id = $2
    print id >> (dir "/order")
    printf "" > (dir "/" id ".old")
    printf "" > (dir "/" id ".new")
    mode = ""
    next
}
/^@@DESCRIPTION / { print substr($0, 15) > (dir "/" id ".description"); next }
/^@@CONVENTION / { print substr($0, 14) > (dir "/" id ".convention"); next }
/^@@FIXTURE /    { print substr($0, 11) > (dir "/" id ".fixture"); next }
/^@@EXPECTATION / { print substr($0, 15) > (dir "/" id ".expectation"); next }
/^@@WHY /        { print substr($0, 7)  > (dir "/" id ".why"); next }
/^@@NOTE /       { print substr($0, 8)  > (dir "/" id ".note"); next }
/^@@OLD/        { mode = "old"; next }
/^@@NEW/        { mode = "new"; next }
{
    if (id != "" && mode != "") print $0 > (dir "/" id "." mode)
}
SPLIT_AWK
printf '%s\n' "$MUT_SPEC" | awk -v dir="$MUT_DIR" -f "$SCRATCH/split.awk"

mapfile -t MUT_IDS < "$MUT_DIR/order"

meta_field() {  # $1 = id, $2 = field name
    [ -f "$MUT_DIR/$1.$2" ] && cat "$MUT_DIR/$1.$2"
}

# -----------------------------------------------------------------------------
# The occurrence count of a MULTI-LINE anchor.
#
# `grep -c` counts matching LINES, not matches, so it cannot answer "does
# this BLOCK occur once?". The count is therefore computed here: every line
# of the source that could START the block is a candidate, and the whole
# block is compared byte for byte at each candidate. The number of exact
# block matches IS the number of occurrences.
#
# Prints "<count> <first start line>" on stdout.
# -----------------------------------------------------------------------------
count_block() {  # $1 = block file, $2 = source file
    local oldf="$1" src="$2"
    local first n count=0 firstline=""
    first="$(head -1 "$oldf")"
    n="$(wc -l < "$oldf")"
    if [ -z "$first" ] || [ "$n" -eq 0 ]; then
        echo "0 0"
        return
    fi
    local ln
    while IFS= read -r ln; do
        [ -z "$ln" ] && continue
        if sed -n "${ln},$((ln + n - 1))p" "$src" | cmp -s - "$oldf"; then
            count=$((count + 1))
            [ -z "$firstline" ] && firstline="$ln"
        fi
    done < <(grep -n -F -x -- "$first" "$src" | cut -d: -f1)
    echo "$count $firstline"
}

# -----------------------------------------------------------------------------
# Apply one mutation: the two-step sed edit with the placeholder token.
# Prints "APPLIED <slot>" on success, or "ANCHOR <count>" / "NOOP <why>".
# -----------------------------------------------------------------------------
apply_mutation() {  # $1 = id
    local id="$1"
    local oldf="$MUT_DIR/$id.old" newf="$MUT_DIR/$id.new"
    local slot="@@MUTATION_SLOT_${id}@@"
    local n start end

    if [ ! -s "$oldf" ]; then
        echo "NOOP the OLD block of $id is empty"
        return
    fi

    # The slot token must not already be in the source: it is the handle the
    # second step works by.
    if grep -q -F -- "$slot" "$SRC_CPP"; then
        echo "NOOP the placeholder token of $id is already in the source"
        return
    fi

    # THE ANCHOR COUNT. More than one occurrence is a coin flip, not a
    # verdict, and the edit is REFUSED.
    local info count
    info="$(count_block "$oldf" "$SRC_CPP")"
    count="${info%% *}"
    start="${info##* }"
    if [ "$count" -ne 1 ]; then
        echo "ANCHOR $count"
        return
    fi

    n="$(wc -l < "$oldf")"
    end=$((start + n - 1))

    # STEP 1: the whole OLD block becomes ONE placeholder line.
    if ! sed -i "${start},${end}c\\${slot}" "$SRC_CPP"; then
        restore
        echo "NOOP step 1 failed"
        return
    fi
    if [ "$(grep -c -F -x -- "$slot" "$SRC_CPP")" -ne 1 ]; then
        restore
        echo "NOOP the placeholder did not land exactly once"
        return
    fi

    # STEP 2: the placeholder line becomes the NEW block, read from a file so
    # that not one character of it has to be escaped.
    {
        echo "/^${slot}\$/ {"
        echo "r ${newf}"
        echo "d"
        echo "}"
    } > "$SCRATCH/step2.sed"
    if ! sed -i -f "$SCRATCH/step2.sed" "$SRC_CPP"; then
        restore
        echo "NOOP step 2 failed"
        return
    fi

    # The edit must have landed, and landed EXACTLY where it was supposed to.
    # The check is byte-exact: the OLD block must be gone, the placeholder
    # must be gone, and the NEW block must be present -- and the source must
    # differ from the pristine backup in EXACTLY those lines.
    if grep -q -F -- "$slot" "$SRC_CPP"; then
        restore
        echo "NOOP the placeholder survived step 2"
        return
    fi
    local oldn newn
    oldn="$(count_block "$oldf" "$SRC_CPP" | cut -d' ' -f1)"
    if [ "$oldn" -ne 0 ]; then
        restore
        echo "NOOP the OLD block is still present"
        return
    fi
    if [ -s "$newf" ]; then
        newn="$(count_block "$newf" "$SRC_CPP" | cut -d' ' -f1)"
        if [ "$newn" -ne 1 ]; then
            restore
            echo "NOOP the NEW block is present $newn times (expected 1)"
            return
        fi
    fi
    # THE BYTE-EXACT CHECK. The mutated source must equal the pristine
    # backup with exactly the OLD block replaced by the NEW block -- no
    # other line may have moved. This is what catches a sed range that was
    # off by one, which a count-based check would not.
    {
        head -n $((start - 1)) "$BAK"
        cat "$newf"
        tail -n +$((end + 1)) "$BAK"
    } > "$SCRATCH/expected.cpp"
    if ! cmp -s "$SCRATCH/expected.cpp" "$SRC_CPP"; then
        restore
        echo "NOOP the mutated source is not the pristine source with the block replaced"
        return
    fi
    echo "APPLIED $slot"
}

# -----------------------------------------------------------------------------
# The build. ONLY the one test binary, out of the current (possibly mutated)
# source, linked against the library objects MINUS the mutated TU's own
# object, which is replaced by the separately compiled mutant.
#
# The test TU is compiled ONCE: no mutation touches the header, so
# recompiling it per mutation would multiply the harness's runtime for no
# extra coverage. The mutated TU IS recompiled every time.
# -----------------------------------------------------------------------------
prepare_tests() {
    rm -f "$MUT_BUILD/$SUITE_NAME.o"
    g++ $CXXFLAGS_LOCAL -c "$TEST_SRC" -o "$MUT_BUILD/$SUITE_NAME.o" \
        2>"$MUT_BUILD/build.log" || return 90
    return 0
}

build_suite() {
    rm -f "$MUT_BUILD/contingency.o" "$MUT_BUILD/$SUITE_NAME"
    g++ $CXXFLAGS_LOCAL -c "$SRC_CPP" -o "$MUT_BUILD/contingency.o" \
        2>"$MUT_BUILD/build.log" || return 90
    if [ ! -f "$MUT_BUILD/$SUITE_NAME.o" ]; then
        prepare_tests || return 90
    fi
    g++ "$MUT_BUILD/$SUITE_NAME.o" "$MUT_BUILD/contingency.o" $LIB_LINK \
        -o "$MUT_BUILD/$SUITE_NAME" 2>>"$MUT_BUILD/build.log" || return 90
    return 0
}

# The suite's verdict is taken from its EXIT CODE and from any [FAIL] lines,
# so that the harness does not have to know the suite's summary format.
run_suite() {
    local out rc
    out=$("$MUT_BUILD/$SUITE_NAME" 2>&1); rc=$?
    LAST_OUT="$out"
    return $rc
}

failed_count() { echo "$1" | grep -c '\[FAIL\]'; }

# -----------------------------------------------------------------------------
# --verify: check every anchor and stop. This is the mode to run first after
# the implementation changes -- a stale anchor is a harness defect and must
# be re-pinned, not scored.
# -----------------------------------------------------------------------------
if [ "$VERIFY_ONLY" -eq 1 ]; then
    bad=0
    for id in "${MUT_IDS[@]}"; do
        info="$(count_block "$MUT_DIR/$id.old" "$SRC_CPP")"
        count="${info%% *}"
        if [ "$count" -eq 1 ]; then
            printf '%-4s ANCHOR-OK   %s\n' "$id" "$(meta_field "$id" description)"
        else
            printf '%-4s ANCHOR-BAD  occurrences=%s  %s\n' "$id" "$count" \
                "$(meta_field "$id" description)"
            bad=$((bad + 1))
        fi
    done
    echo ""
    echo "anchors: ${#MUT_IDS[@]} mutations, $bad with a non-unique or missing anchor"
    [ "$bad" -eq 0 ] || exit 1
    exit 0
fi

# -----------------------------------------------------------------------------
# The pristine self-check: the suite must be GREEN before any mutation runs.
# A harness that starts from a red tree reports garbage.
# -----------------------------------------------------------------------------
if [ ! -f "$TEST_SRC" ]; then
    echo "HARNESS DEFECT: $TEST_SRC does not exist. The suite must be in place" >&2
    echo "before this harness can say anything about it." >&2
    exit 99
fi
if ! prepare_tests; then
    echo "HARNESS DEFECT: the test translation unit does not compile." >&2
    tail -20 "$MUT_BUILD/build.log" >&2
    exit 99
fi
if ! build_suite; then
    echo "HARNESS DEFECT: the pristine tree does not build." >&2
    tail -20 "$MUT_BUILD/build.log" >&2
    exit 99
fi
PRISTINE_OK=1
run_suite || PRISTINE_OK=0
nf="$(failed_count "$LAST_OUT")"
if [ "$PRISTINE_OK" -ne 1 ] || [ "$nf" -ne 0 ]; then
    echo "HARNESS DEFECT: the pristine suite is not green (${nf:-no summary} failures)." >&2
    echo "$LAST_OUT" | tail -20 >&2
    exit 99
fi
echo "pristine: $SUITE_NAME -- 0 failures"
echo "pristine: green -- mutation may begin"
# From here on the harness mutates the source, so the EXIT trap must rebuild
# and re-verify the RESTORED tree before it reports anything.
POST_VERIFY=1

# The verdict table. The separator is the ASCII unit separator, because a
# description or a failure message may contain a vertical bar.
US=$'\x1f'
CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0; ANCHOR=0; INERT=0
declare -a TABLE

record() {  # $1 = id, $2 = verdict, $3 = detail
    TABLE+=("$1$US$2$US$3")
}

verdict_of() {  # $1 = id
    local row
    for row in "${TABLE[@]}"; do
        case "$row" in
            "$1$US"*) echo "${row#"$1$US"}" | cut -d"$US" -f1; return ;;
        esac
    done
    echo ""
}

detail_of() {  # $1 = id
    local row
    for row in "${TABLE[@]}"; do
        case "$row" in
            "$1$US"*) echo "${row#"$1$US"}" | cut -d"$US" -f2-; return ;;
        esac
    done
    echo ""
}

print_table() {
    printf '%-5s %-11s %-52s %-12s %s\n' "ID" "CONV" "MUTATION" "FIXTURE" "VERDICT"
    printf '%-5s %-11s %-52s %-12s %s\n' "----" "-----------" \
        "----------------------------------------------------" "------------" "-------"
    local id v d
    for id in "${MUT_IDS[@]}"; do
        v="$(verdict_of "$id")"
        d="$(detail_of "$id")"
        printf '%-5s %-11s %-52s %-12s %s\n' "$id" \
            "$(meta_field "$id" convention)" "$(meta_field "$id" description)" \
            "$(meta_field "$id" fixture)" "$v"
        [ -n "$d" ] && printf '%-5s %-11s %-52s %-12s %s\n' "" "" "" "" "$d"
    done
}

echo
echo "=== contingency mutations ==="

for id in "${MUT_IDS[@]}"; do
    restore
    desc="$(meta_field "$id" description)"
    expect="$(meta_field "$id" expectation)"
    why="$(meta_field "$id" why)"

    before="$(checksum "$SRC_CPP")"
    out="$(apply_mutation "$id")"
    after="$(checksum "$SRC_CPP")"

    case "$out" in
        "ANCHOR "*)
            record "$id" "MUT-ANCHOR" "the anchor occurs ${out#ANCHOR } times -- a coin flip is not a verdict; the edit was REFUSED"
            ANCHOR=$((ANCHOR + 1))
            echo "$id MUT-ANCHOR  the anchor is not unique -- harness defect, not a verdict"
            continue ;;
        "NOOP "*)
            record "$id" "MUT-NOOP" "${out#NOOP } -- the edit did not land; harness defect, not a verdict"
            NOOP=$((NOOP + 1))
            echo "$id MUT-NOOP  ${out#NOOP } -- harness defect, not a verdict"
            continue ;;
    esac

    if [ "$before" = "$after" ]; then
        record "$id" "MUT-NOOP" "the source's checksum did not change -- harness defect, not a verdict"
        NOOP=$((NOOP + 1))
        echo "$id MUT-NOOP  the source did not change -- harness defect, not a verdict"
        continue
    fi

    build_suite
    rc=$?
    if [ "$rc" -eq 90 ]; then
        record "$id" "MUT-BROKEN" "the build failed -- NOT a verdict"
        BROKEN=$((BROKEN + 1))
        echo "$id MUT-BROKEN  the build failed -- NOT a verdict"
        restore
        continue
    fi

    run_suite; rc=$?
    suite_out="$LAST_OUT"
    nfail="$(failed_count "$suite_out")"

    if [ "$rc" -ge 128 ]; then
        record "$id" "MUT-CRASH" "the binary died with signal status $rc -- NOT a verdict"
        CRASH=$((CRASH + 1))
        echo "$id MUT-CRASH  the binary died with status $rc -- NOT a verdict"
        restore
        continue
    fi

    if [ "$rc" -ne 0 ] || [ "$nfail" -ne 0 ]; then
        line="$(echo "$suite_out" | grep -m1 '\[FAIL\]' | sed 's/^ *//' | cut -c1-96)"
        record "$id" "CAUGHT" "$line"
        CAUGHT=$((CAUGHT + 1))
        echo "$id CAUGHT  $desc  [$nfail failures]"
        [ -n "$line" ] && echo "      $line"
    else
        if [ "$expect" = "inert" ]; then
            record "$id" "SURVIVED" "declared INERT: $why"
            INERT=$((INERT + 1))
            echo "$id SURVIVED  $desc  (declared INERT: $why)"
        else
            record "$id" "SURVIVED" "the suite stayed green -- a REAL coverage gap in the fixture"
            SURVIVED=$((SURVIVED + 1))
            echo "$id SURVIVED  $desc  -- the suite stayed green: a REAL coverage gap"
            echo "      does any fixture vary in this dimension? (the one that must is $(meta_field "$id" fixture))"
            note="$(meta_field "$id" note)"
            [ -n "$note" ] && echo "      NOTE: $note"
        fi
    fi
    restore
done

echo
echo "=== caught $CAUGHT / survived $SURVIVED / declared-inert $INERT / broken $BROKEN / crashed $CRASH / noop $NOOP / anchor $ANCHOR ==="
echo ""
print_table
echo ""
echo "FINAL: $CAUGHT caught / $SURVIVED survived / $INERT declared-inert / $BROKEN broken / $CRASH crashed / $NOOP noop / $ANCHOR anchor"

if [ "$SURVIVED" -gt 0 ]; then
    echo "NOTE: $SURVIVED unexplained survivor(s) -- these are REAL coverage gaps." >&2
    exit 3
fi
if [ "$STRICT" -eq 1 ] && [ "$INERT" -gt 0 ]; then
    echo "NOTE: --strict was given, so the $INERT declared-inert survivor(s) fail the run too." >&2
    exit 3
fi
if [ "$BROKEN" -gt 0 ] || [ "$CRASH" -gt 0 ]; then
    echo "NOTE: $BROKEN broken and $CRASH crashed mutation(s) -- re-run in isolation." >&2
    exit 1
fi
if [ "$NOOP" -gt 0 ] || [ "$ANCHOR" -gt 0 ]; then
    echo "NOTE: $NOOP noop and $ANCHOR anchor harness defect(s)." >&2
    exit 1
fi
exit 0
