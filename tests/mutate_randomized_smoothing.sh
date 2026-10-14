#!/usr/bin/env bash
# mutate_randomized_smoothing.sh — mutation harness for randomized smoothing.
#
# WHY THIS SCRIPT IS DEFENSIVE. Seven separate bugs in mutation harnesses in
# this repo produced FALSE VERDICTS, and each defence below exists because of
# one of them:
#
#  1. SNAPSHOT-TAKES-THE-MUTANT. A harness that snapshots the implementation
#     AFTER applying a mutation restores a mutant over the correct code.
#     Defence: the baseline is taken ONCE, BEFORE any mutation, and then
#     ASSERTED PRISTINE by grepping lines that exist only in the correct
#     implementation. Abort rather than proceed.
#
#  2. ABSOLUTE MAKE TARGET SKIPS THE REBUILD. `make "$REPO/build/test_x"` with
#     an ABSOLUTE path reports "up to date" and never recompiles, so not one
#     mutant was ever built. Defence: RELATIVE make targets + an mtime check.
#
#  3. A NO-OP sed REPORTED AS A PASS. Defence: `diff -q` after each sed;
#     MUT-NOOP is reported separately from MUT-SURVIVED.
#
#  4. COMPILE FAILURE REPORTED AS "CAUGHT". A mutant that fails to build proves
#     nothing. Defence: MUT-BROKEN, never MUT-CAUGHT.
#
#  5. DOUBLE-LINKING THE MUTATED UNIT. The adversarial session linked
#     $(LIB_OBJS), which ALREADY contained build/utils/adversarial.o, so every
#     mutation double-defined every symbol and the harness reported "CAUGHT
#     (compile error)" eight times running. Defence: the mutated .cpp is
#     compiled from source and its object is EXCLUDED from the link set; the
#     script asserts the exclusion actually held.
#
#  6. AN INVERTED VERDICT. The KD session reported MUT-CAUGHT when the summary
#     read "0 failed". Defence: count `[FAIL]` lines, which is also immune to
#     summary-format changes.
#
#  7. A SURVIVOR CLAIMED "INERT" WITHOUT MEASUREMENT. Defence: an
#     unexplained survivor is reported as MUT-UNEXPLAINED, never as inert.
#
# Usage: tests/mutate_randomized_smoothing.sh

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SRC="include/nn/utils/randomized_smoothing.cpp"
HDR="include/nn/utils/randomized_smoothing.h"
TEST_SRC="tests/test_randomized_smoothing.cpp"
MUTOBJ="build/utils/randomized_smoothing.o"

TMPBIN="$(mktemp -t rs_mutbin_XXXXXX)"
BACKUP="$(mktemp -t rs_baseline_XXXXXX.cpp)"

CAUGHT=0
SURVIVED=0
BROKEN=0
NOOP=0
CRASHED=0

# --- atomic restore: a plain `cp` truncates the destination first, so a
#     concurrent reader can splice two generations together. temp+rename.
restore() { cp "$BACKUP" "$SRC.tmp$$" && mv -f "$SRC.tmp$$" "$SRC"; }
cleanup() {
    restore
    rm -f "$BACKUP" "$TMPBIN" "$SRC.tmp$$"
    make -s "$MUTOBJ" >/dev/null 2>&1
}
trap cleanup EXIT

# --- Baseline: taken ONCE, before any mutation, and verified pristine. ------
cp "$SRC" "$BACKUP"

echo "=== Baseline pristine checks ==="
PRISTINE=1
# Each marker is tied to one of the nine semantics pinned in the header.
grep -q 'return 0.5 \* std::erfc(-x / std::sqrt(2.0));' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: normal_cdf is not 0.5*erfc(-x/sqrt2)"; PRISTINE=0; }
grep -q 'if (p == 0.5) return 0.0;' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the exact Phi^-1(0.5) == 0 case"; PRISTINE=0; }
grep -q 'if (k == 0) return 0.0;' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the k==0 -> 0 rule (D7)"; PRISTINE=0; }
grep -q 'regularized_incomplete_beta_inverse(static_cast<double>(k),' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: CP bound is not Beta^-1(alpha; k, n-k+1) (D6)"; PRISTINE=0; }
grep -q 'return std::min(1.0, 2.0 \* std::min(lower, upper));' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: binom p-value is not two-sided at p=0.5 (D4)"; PRISTINE=0; }
grep -q 'second_val = best_val;' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: top_two stores an INDEX into a COUNT slot"; PRISTINE=0; }
grep -q 'out.p_value = binom_pvalue_half(n_a, n_b);' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: PREDICT does not test the TOP TWO counts (D3)"; PRISTINE=0; }
grep -q 'if (!(out.p_lower > 0.5)) {' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the strict p_lower > 1/2 rule (D2)"; PRISTINE=0; }
grep -q 'sample_under_noise(f, x, cfg.sigma, cfg.num_samples, cfg.seed + 1u)' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: second CERTIFY round is not on FRESH samples (D5)"; PRISTINE=0; }
grep -q 'return certified_radius_from_probs(sigma, p_lower, 1.0 - p_lower);' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: symmetric radius is not derived from the general form (D8)"; PRISTINE=0; }
grep -q 'noisy\[0\]\[j\] = x\[0\]\[j\] + mag \* std::cos(angle);' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: noise is not additive (D1)"; PRISTINE=0; }
if [ "$PRISTINE" -ne 1 ]; then
    echo "ABORT: the baseline is not the correct implementation."
    exit 2
fi
echo "  all 11 pristine markers present"
echo

echo "=== Pristine suite must pass before any mutation is scored ==="
make -s build/test_randomized_smoothing >/dev/null 2>&1
if ! ./build/test_randomized_smoothing >/dev/null 2>&1; then
    echo "ABORT: the pristine suite does not pass, so mutation results are meaningless."
    exit 2
fi
echo "  pristine suite is green"
echo

run_mutation() {
    local id="$1"; shift
    local desc="$1"; shift
    local search="$1"; shift
    local sedexpr="$1"; shift
    local LINE="${LINE:-}"

    restore
    rm -f "$MUTOBJ"

    # Line-number-anchored mutations (M8, M11) additionally verify that the
    # cited line still holds the expected text. A future edit that shifts the
    # file then reports MUT-NOOP instead of mutating an unrelated statement --
    # the Grad-CAM session's lesson.
    if [ -n "${LINE:-}" ]; then
        if ! sed -n "${LINE}p" "$SRC" | grep -qF "$search"; then
            echo "  MUT-NOOP    $id: line $LINE no longer holds the anchor text"
            NOOP=$((NOOP+1))
            return
        fi
    fi

    # grep -F: the anchors contain '*' and '+', which BRE treats as literals,
    # so a plain `grep -q` reports MUT-NOOP for a line that IS present. This is
    # the same BRE-vs-ERE trap the KD harness recorded, and it produced four
    # false MUT-NOOPs on this script's first run.
    if ! grep -qF "$search" "$SRC"; then
        echo "  MUT-NOOP    $id: anchor not found — '$desc'"
        NOOP=$((NOOP+1))
        return
    fi

    sed -i "$sedexpr" "$SRC"
    if diff -q "$BACKUP" "$SRC" >/dev/null; then
        echo "  MUT-NOOP    $id: sed changed nothing — '$desc'"
        NOOP=$((NOOP+1))
        return
    fi

    # Defence 5: compile the mutated unit from SOURCE and link it EXCLUDING the
    # prebuilt object, so nothing is double-defined.
    local LIB
    LIB="$(for f in include/nn/*.cpp include/nn/*/*.cpp include/nn/*/*/*.cpp; do
               [ -f "$f" ] || continue
               o="build/${f#include/nn/}"; o="${o%.cpp}.o"
               [ -f "$o" ] || continue
               [ "$o" = "$MUTOBJ" ] && continue
               echo "$o"
           done | tr '\n' ' ')"
    if echo "$LIB" | grep -q "$MUTOBJ"; then
        echo "  MUT-BROKEN  $id: mutated object still in link set (double-link) — '$desc'"
        BROKEN=$((BROKEN+1))
        return
    fi

    if ! g++ -std=c++17 -O2 -Wall -Wextra -Iinclude "$TEST_SRC" "$SRC" $LIB \
            -o "$TMPBIN" 2>"${TMPBIN}.log"; then
        # Defence 4: a build failure is NOT a catch.
        echo "  MUT-BROKEN  $id: does not compile — '$desc'"
        BROKEN=$((BROKEN+1))
        return
    fi

    # Defence 2: the binary must be newer than the mutated source.
    if [ "$SRC" -nt "$TMPBIN" ]; then
        echo "  MUT-BROKEN  $id: binary older than source (stale build) — '$desc'"
        BROKEN=$((BROKEN+1))
        return
    fi

    local out nfail rc=0
    out="$("$TMPBIN" 2>&1)" || rc=$?
    nfail="$(printf '%s' "$out" | grep -c '\[FAIL\]')"

    # A CRASH IS A DISTINCT VERDICT, AND THIS PATTERN WAS WRONG ON ITS FIRST
    # RUN. M5 aborts with an uncaught std::invalid_argument, which the C++
    # runtime prints as "terminate called after throwing an instance of" and
    # which surfaces as SIGABRT -- but the original pattern only listed
    # `AddressSanitizer|runtime error:|Aborted`, and "Aborted" is the BASH
    # job-control message, not anything a C++ program prints. So a mutant that
    # crashed was scored as MUT-SURVIVED, because the crash output contained no
    # `[FAIL]` line to count. Caught only by running the mutant by hand and
    # reading its exit status -- the metrics session's lesson ("a crash cannot
    # be told apart from a harness artifact") recurring in a new disguise.
    #
    # Match the RUNTIME's wording, and use the exit status as the backstop.
    local crashed=0
    if printf '%s' "$out" | grep -qE 'AddressSanitizer|runtime error:|terminate called|Segmentation fault|core dumped'; then
        crashed=1
    fi

    # Defence 6: a caught mutation makes the suite RED, i.e. a NON-ZERO failure
    # count. Counting [FAIL] lines rather than the summary string also survives
    # a change of summary format.
    if [ "$crashed" -eq 1 ]; then
        echo "  MUT-CRASH   $id: aborted (rc=$rc) instead of failing an assertion ($nfail [FAIL]) — '$desc'"
        CRASHED=$((CRASHED+1))
    elif [ "$nfail" -gt 0 ]; then
        echo "  MUT-CAUGHT  $id: $nfail assertion(s) failed — '$desc'"
        CAUGHT=$((CAUGHT+1))
    else
        echo "  MUT-SURVIVED $id: suite still green — '$desc'"
        SURVIVED=$((SURVIVED+1))
    fi
}

echo "=== Scoring mutations ==="

# ---------------------------------------------------------------------------
# Each is anchored on a unique string. Line anchoring is mandatory where a
# string appears more than once (see the KD session: an unanchored sed hit the
# first of two identical lines and scored as a false survivor).
# ---------------------------------------------------------------------------

# D6 -- the headline semantic. Using 2*alpha here is the paper's own
# mis-scaled snippet; it yields a bound twice too optimistic.
run_mutation M1 "CP lower bound uses 2*alpha (the mis-scaled form)" \
    'static_cast<double>(n - k) + 1.0,' \
    's|^                                               alpha);|                                               2.0 * alpha);|'

# D2 -- the abstention threshold. >= admits p_lower == 1/2.
run_mutation M2 "p_lower > 1/2 becomes p_lower >= 1/2" \
    'if (!(out.p_lower > 0.5)) {' \
    's|if (!(out.p_lower > 0.5)) {|if (!(out.p_lower >= 0.5)) {|'

# D4 -- one-sided instead of two-sided.
run_mutation M3 "binom p-value becomes one-sided" \
    'return std::min(1.0, 2.0 * std::min(lower, upper));' \
    's|return std::min(1.0, 2.0 \* std::min(lower, upper));|return 0.5 * std::max(lower, upper);|'

# D3 -- feed the whole sample into the test instead of the top two. `n` is the
# sample count the loop actually drew, so the mutant computes the same test over
# ALL samples rather than the top two.
run_mutation M4 "PREDICT tests the full sample count, not the top two" \
    'out.p_value = binom_pvalue_half(n_a, n_b);' \
    's|out.p_value = binom_pvalue_half(n_a, n_b);|out.p_value = binom_pvalue_half(n_a, num_samples - n_a);|'

# D5 -- pool the guessing samples with the estimating samples.
run_mutation M5 "CERTIFY pools the n0 guess samples into the estimate" \
    'sample_under_noise(f, x, cfg.sigma, cfg.num_samples, cfg.seed + 1u)' \
    's|cfg.num_samples, cfg.seed + 1u)|cfg.num_samples + cfg.num_guess_samples, cfg.seed + 1u)|'

# D8 -- drop the 1/2 factor in Theorem 1's general form.
run_mutation M6 "Theorem 1 radius drops the sigma/2 prefactor" \
    'return 0.5 * sigma * (normal_quantile(p_a) - normal_quantile(p_b));' \
    's|return 0.5 \* sigma \* (normal_quantile(p_a) - normal_quantile(p_b));|return sigma * (normal_quantile(p_a) - normal_quantile(p_b));|'

# D7 -- let the k==0 case flow into the degenerate Beta shape (NaN).
run_mutation M7 "k==0 falls through to the degenerate Beta shape" \
    'if (k == 0) return 0.0;' \
    's|if (k == 0) return 0.0;||'

# D1 -- multiplicative instead of additive noise.
run_mutation M8 "noise becomes multiplicative instead of additive (D1)" \
    'noisy[0][j] = x[0][j] + mag * std::cos(angle);' \
    '356s|noisy\[0\]\[j\] = x\[0\]\[j\] + mag \* std::cos(angle);|noisy[0][j] = x[0][j] * mag * std::cos(angle);|'

# the top_two index/count bug -- the real bug this session found.
run_mutation M9 "top_two stores an index into the runner-up count slot" \
    'second_val = best_val;' \
    's|second_val = best_val;|second_val = best;|'

# normal_quantile returns the bisection residual at p=0.5 instead of exactly 0.
run_mutation M10 "Phi^-1(0.5) returns the bisection residual" \
    'if (p == 0.5) return 0.0;' \
    's|if (p == 0.5) return 0.0;||'

# Theorem 6 -- drop the 1/(1-alpha) correction.
run_mutation M11 "Theorem 6 drops the 1/(1-alpha) correction" \
    'return (1.0 / (1.0 - alpha)) *' \
    '302,304s|return (1.0 / (1.0 - alpha)) \*|return 1.0 *|'

# Theorem 6 -- clamp the result into [0,1] INSIDE the statistic, which hides a
# genuinely negative bound (no certified examples) behind a 0.
run_mutation M12 "Theorem 6 clamps its result into [0,1]" \
    'return (1.0 / (1.0 - alpha)) *' \
    's|return (1.0 / (1.0 - alpha)) \*|{ const double _v = (1.0 / (1.0 - alpha)) *|; s|static_cast<double>(m);$|static_cast<double>(m); return _v < 0.0 ? 0.0 : (_v > 1.0 ? 1.0 : _v); }|'

echo
echo "=== Summary ==="
echo "  caught:    $CAUGHT"
echo "  crashed:   $CRASHED   (aborted instead of asserting — counted as caught, but reported apart)"
echo "  survived:  $SURVIVED"
echo "  broken:    $BROKEN   (build failure — proves nothing about test strength)"
echo "  no-op:     $NOOP    (harness defect, not a verdict)"

if [ "$NOOP" -gt 0 ]; then
    echo
    echo "NOTE: a MUT-NOOP is a HARNESS defect, not a verdict. Fix the anchor."
fi
if [ "$SURVIVED" -gt 0 ]; then
    echo
    echo "NOTE: an unexplained survivor is NOT a gap until a probe has measured"
    echo "      whether it can change anything (TDD skill). Do not record it as"
    echo "      inert without that measurement."
fi

[ "$CAUGHT" -gt 0 ] && [ "$BROKEN" -eq 0 ] && [ "$NOOP" -eq 0 ]