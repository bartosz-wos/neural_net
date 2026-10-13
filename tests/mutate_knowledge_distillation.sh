#!/usr/bin/env bash
# mutate_knowledge_distillation.sh — mutation harness for knowledge distillation.
#
# WHY THIS SCRIPT IS DEFENSIVE. Four separate bugs in mutation harnesses in this
# repo produced FALSE VERDICTS, and each defence below exists because of one:
#
#  1. SNAPSHOT-TAKES-THE-MUTANT. A harness that snapshots the implementation
#     AFTER applying a mutation restores a mutant over the correct code. Every
#     later "restore" silently reintroduces the bug and the suite stays green.
#     Defence: the baseline is taken ONCE, BEFORE any mutation, and then
#     ASSERTED PRISTINE by grepping for lines that exist only in the correct
#     implementation. Abort rather than proceed.
#
#  2. ABSOLUTE MAKE TARGET SKIPS THE REBUILD. `make "$REPO/build/test_x"` with
#     an ABSOLUTE path reports "up to date" and never recompiles, so not one
#     mutant was ever built and every mutation reported MUT-SURVIVED.
#     Defence: RELATIVE make targets only, plus an mtime check that rejects any
#     run whose binary is older than the mutated source.
#
#  3. A NO-OP sed REPORTED AS A PASS. If the mutation string is not present in
#     the file, sed changes nothing and the (unmutated) suite passes, which is
#     indistinguishable from "the mutation was caught".
#     Defence: `diff -q` after each sed; report MUT-NOOP separately.
#
#  4. COMPILE FAILURE REPORTED AS "CAUGHT". A mutant that fails to build proves
#     nothing about test strength.
#     Defence: compile failure is MUT-BROKEN, never MUT-CAUGHT.
#
#  5. NEVER run `make` against the same tree while a mutation harness is running.
#     The build caches the mutant into a .o and later links the broken code.
#     Defence: this script builds ONE binary from explicitly listed sources, so
#     there is no shared .o for an unrelated make to poison.
#
# Usage: tests/mutate_knowledge_distillation.sh

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SRC="include/nn/utils/knowledge_distillation.cpp"
TEST_SRC="tests/test_knowledge_distillation.cpp"
TMPBIN="$(mktemp -t kd_mutbin_XXXXXX)"
BACKUP="$(mktemp -t kd_baseline_XXXXXX.cpp)"

CAUGHT=0
SURVIVED=0
BROKEN=0
NOOP=0

cleanup() {
    cp "$BACKUP" "$SRC"
    rm -f "$BACKUP" "$TMPBIN"
}
trap cleanup EXIT

# --- Baseline: taken ONCE, before any mutation, and verified pristine. -------
cp "$SRC" "$BACKUP"

echo "=== Baseline pristine checks ==="
PRISTINE=1
# Lines that exist ONLY in the correct implementation. Each is tied to one of
# the nine semantics pinned in the header.
grep -q 'dlp(i, j) = -scale \* p_t(i, j);' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the NEGATIVE adjoint sign (-scale*p_t)"; PRISTINE=0; }
grep -q 'return value \* T \* T;' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the T^2 loss rescaling"; PRISTINE=0; }
grep -q 'if (p == 0.0) continue;' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the xlogy zero rule"; PRISTINE=0; }
grep -q 'value /= static_cast<double>(student_logits.rows \* student_logits.cols);' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the mean-over-ELEMENTS divisor"; PRISTINE=0; }
grep -q 'log_softmax(student_logits \* (1.0 / T))' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the DIVIDE-by-T in student_log_probs"; PRISTINE=0; }
grep -q 'hard_ce_adjoint(student_logits, labels,' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the hard-CE adjoint call"; PRISTINE=0; }
grep -q 'alpha_ \* hard_row_scale' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: hard adjoint is not weighted by alpha"; PRISTINE=0; }
if [ "$PRISTINE" -ne 1 ]; then
    echo "ABORT: the baseline is not the correct implementation."
    exit 2
fi
echo "  all 7 pristine markers present"
echo

# The suite must pass on the pristine baseline before any mutation is scored.
make -s "build/test_knowledge_distillation" >/dev/null 2>&1
if ! ./build/test_knowledge_distillation >/dev/null 2>&1; then
    echo "ABORT: the pristine suite does not pass, so mutation results would be meaningless."
    exit 2
fi
echo "=== Pristine suite passes; scoring mutations ==="
echo

run_mutation() {
    local id="$1"; shift
    local desc="$1"; shift
    # Remaining args: sed expression(s) passed to apply_mutations.
    local sedexpr="$1"; shift
    local search="${1:-}"

    cp "$BACKUP" "$SRC"

    if [ -n "$search" ]; then
        if ! grep -q "$search" "$SRC"; then
            echo "  MUT-NOOP    $id: anchor not found in source — '$desc'"
            NOOP=$((NOOP+1))
            return
        fi
    fi

    sed -i "$sedexpr" "$SRC"
    if diff -q "$BACKUP" "$SRC" >/dev/null; then
        echo "  MUT-NOOP    $id: sed changed nothing — '$desc'"
        NOOP=$((NOOP+1))
        return
    fi

    # Build from explicit sources: no shared .o for anything else to poison.
    if ! g++ -std=c++17 -O2 -Iinclude "$TEST_SRC" "$SRC" \
            include/nn/utils/losses.cpp include/nn/core/tensor.cpp \
            -o "$TMPBIN" 2>/dev/null; then
        echo "  MUT-BROKEN  $id: does not compile — '$desc'"
        BROKEN=$((BROKEN+1))
        return
    fi

    # mtime guard: the binary must be newer than the mutated source.
    if [ "$SRC" -nt "$TMPBIN" ]; then
        echo "  MUT-BROKEN  $id: binary older than source (stale build) — '$desc'"
        BROKEN=$((BROKEN+1))
        return
    fi

    local out
    out="$("$TMPBIN" 2>&1)"
    local nfail
    nfail="$(printf '%s' "$out" | grep -c '\[FAIL\]')"

    # CORRECTION: the first version of this script had the verdict INVERTED —
    # it reported MUT-CAUGHT when the summary matched "0 failed" and
    # MUT-SURVIVED otherwise, so five mutations that genuinely failed 13, 20, 1
    # and 14 assertions respectively were reported as survivors. A caught
    # mutation is one that makes the suite RED, so the test is for a non-zero
    # failure count. Counting [FAIL] lines rather than grepping the summary also
    # sidesteps summary-format changes.
    if [ "$nfail" -gt 0 ]; then
        echo "  MUT-CAUGHT  $id: $nfail assertion(s) failed — '$desc'"
        CAUGHT=$((CAUGHT+1))
    else
        echo "  MUT-SURVIVED $id: suite still green — '$desc'"
        SURVIVED=$((SURVIVED+1))
    fi
}

# ---------------------------------------------------------------------------
# Mutations. Each is line-anchored on a unique string; `sed '0,/X/` was already
# shown to mutate the FIRST match in the file and this one has two identical
# lines in two different functions, so anchoring is mandatory.
# ---------------------------------------------------------------------------

run_mutation M1 "drop the T^2 loss rescaling" \
    's|return value \* T \* T;|return value;|' \
    'return value \* T \* T;'

run_mutation M2 "drop the 1/T chain factor in the backward" \
    's|const double net = T \* T \* (1.0 / T);|const double net = T * T;|' \
    'const double net = T \* T \* (1.0 / T);'

run_mutation M3 "drop the adjoint sign (the sign-flip fingerprint)" \
    's|dlp(i, j) = -scale \* p_t(i, j);|dlp(i, j) = scale * p_t(i, j);|' \
    'dlp(i, j) = -scale \* p_t(i, j);'

run_mutation M4 "drop the soft-adjoint sign in DistillationLoss::backward" \
    's|dlp(i, j) = -soft_element_scale \* p_t(i, j);|dlp(i, j) = soft_element_scale * p_t(i, j);|' \
    'dlp(i, j) = -soft_element_scale \* p_t(i, j);'

run_mutation M5 "drop the xlogy zero rule (NaN on underflow)" \
    's|if (p == 0.0) continue;|if (p < -1.0) continue;|' \
    'if (p == 0.0) continue;'

run_mutation M6 "divide the Mean by rows instead of elements (batchmean)" \
    's|value /= static_cast<double>(student_logits.rows \* student_logits.cols);|value /= static_cast<double>(student_logits.rows);|' \
    'value /= static_cast<double>(student_logits.rows \* student_logits.cols);'

run_mutation M7 "drop the alpha weight on the hard CE adjoint" \
    's|alpha_ \* hard_row_scale|hard_row_scale|' \
    'alpha_ \* hard_row_scale'

run_mutation M8 "multiply by T instead of dividing (the inverted op)" \
    's|return log_softmax(student_logits \* (1.0 / T));|return log_softmax(student_logits * T);|' \
    'return log_softmax(student_logits \* (1.0 / T));'

# LINE-ANCHORED, because the identical line occurs TWICE in the file: the
# forward (line 84) and the backward (line 133). An unanchored sed would hit
# both and score as one mutation rather than the two the name implies, and the
# repo has already been bitten by exactly this (`sed '0,/X/'` mutating the first
# of two identical lines and reporting a false MUT-SURVIVED). Targeting the
# backward's own site is what makes this a KL-direction mutation rather than a
# "break both directions" mutation, which any asymmetric fixture would also
# catch and which would therefore prove nothing about the backward specifically.
#
# Handled by an explicit block rather than run_mutation(), because run_mutation()
# applies the sed to EVERY matching line while this must hit exactly one.

# A line-number mutation is only safe if the line it targets still holds the
# expected text. Verify that explicitly rather than trusting the number: if an
# edit above shifts the file, the sed silently mutates a DIFFERENT statement and
# the harness would report a catch for a mutation nobody wrote.
if ! sed -n '133p' "$SRC" | grep -q 'softmax_at_temperature(teacher_logits, T)'; then
    echo "  MUT-NOOP    M9: line 133 no longer holds the backward's teacher softmax"
    echo "               ($(sed -n '133p' "$SRC" | head -c 60))"
    NOOP=$((NOOP+1))
else
    cp "$BACKUP" "$SRC"
    sed -i '133s|teacher_logits|student_logits|' "$SRC"
    if diff -q "$BACKUP" "$SRC" >/dev/null; then
        echo "  MUT-NOOP    M9: sed changed nothing"
        NOOP=$((NOOP+1))
    elif ! g++ -std=c++17 -O2 -Iinclude "$TEST_SRC" "$SRC" \
            include/nn/utils/losses.cpp include/nn/core/tensor.cpp \
            -o "$TMPBIN" 2>/dev/null; then
        echo "  MUT-BROKEN  M9: does not compile"
        BROKEN=$((BROKEN+1))
    else
        out="$("$TMPBIN" 2>&1)"
        nfail="$(printf '%s' "$out" | grep -c '\[FAIL\]')"
        # Same (inverted) verdict bug as run_mutation(), fixed the same way: a
        # caught mutation is one that makes the suite RED.
        if [ "$nfail" -gt 0 ]; then
            echo "  MUT-CAUGHT  M9: $nfail assertion(s) failed — 'swap the KL direction in the backward (student becomes p)'"
            CAUGHT=$((CAUGHT+1))
        else
            echo "  MUT-SURVIVED M9: suite still green — 'swap the KL direction in the backward (student becomes p)'"
            SURVIVED=$((SURVIVED+1))
        fi
    fi
fi

# ---------------------------------------------------------------------------
cp "$BACKUP" "$SRC"

echo
echo "=== Mutation summary ==="
echo "  caught   : $CAUGHT"
echo "  survived : $SURVIVED"
echo "  broken   : $BROKEN"
echo "  no-op    : $NOOP"

if [ "$NOOP" -ne 0 ] || [ "$BROKEN" -ne 0 ]; then
    echo "VERDICT: INCONCLUSIVE — a no-op or broken mutation scores nothing."
    exit 1
fi
if [ "$SURVIVED" -ne 0 ]; then
    echo "VERDICT: a mutation survived. Either the test is too weak (add a"
    echo "         fixture) or the mutation is provably inert (then MEASURE that"
    echo "         with a differential probe, do not assume it)."
    exit 1
fi
echo "VERDICT: every mutation was caught — the suite has teeth."