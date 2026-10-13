#!/usr/bin/env bash
# mutate_temperature_scaling.sh — mutation harness for the temperature scaling
# implementation.
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
#     mutant was ever built and all seven mutations reported MUT-SURVIVED.
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
# Usage: tests/mutate_temperature_scaling.sh

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SRC="include/nn/utils/temperature_scaling.cpp"
BIN="build/test_temperature_scaling"
BACKUP="$(mktemp -t ts_baseline_XXXXXX.cpp)"
TMPBIN="$(mktemp -t ts_mutbin_XXXXXX)"

CAUGHT=0
SURVIVED=0
BROKEN=0
NOOP=0
FAILED_MUTATIONS=()

cleanup() {
    cp "$BACKUP" "$SRC"
    rm -f "$BACKUP" "$TMPBIN"
}
trap cleanup EXIT

# --- Baseline: taken ONCE, before any mutation, and verified pristine. -------
cp "$SRC" "$BACKUP"

echo "=== Baseline pristine checks ==="
PRISTINE=1
grep -q 'logits(i, j) / T' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the DIVIDE 'logits(i, j) / T'"; PRISTINE=0; }
grep -q 'kGoldenRatio = 0.6180339887498949' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing kGoldenRatio"; PRISTINE=0; }
grep -q 'logits(i, y) - pz' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the gradient bracket 'logits(i, y) - pz'"; PRISTINE=0; }
grep -q 'double best_T = t_min;' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing the endpoint-initialised best_T"; PRISTINE=0; }
grep -q 'const double inv_T2 = 1.0 / (T \* T)' "$BACKUP" \
    || { echo "  BASELINE CORRUPT: missing inv_T2"; PRISTINE=0; }
if [ "$PRISTINE" -ne 1 ]; then
    echo "ABORT: the baseline is not the correct implementation."
    exit 1
fi
echo "  baseline is pristine (5 unique markers found)"

run_suite() {
    # RELATIVE make target — an absolute one silently skips the rebuild.
    make "$BIN" >/dev/null 2>&1
    # mtime guard: the binary must be NEWER than the (mutated) source.
    if [ ! -f "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
        echo "  MUT-STALE: binary older than source; the mutation was never built"
        return 2
    fi
    ./"$BIN" >/dev/null 2>&1
    return $?
}

echo
echo "=== Sanity: unmutated suite must pass ==="
if run_suite; then
    echo "  unmutated suite PASSES (good — otherwise every result below is noise)"
else
    rc=$?
    if [ "$rc" -eq 2 ]; then echo "  ABORT: stale binary on the unmutated run"; exit 1; fi
    echo "  ABORT: the unmutated suite FAILS; fix that before mutation testing"
    exit 1
fi

# run_mutation <name> <sed-expr>
run_mutation() {
    local name="$1"; shift
    local expr="$1"; shift
    cp "$BACKUP" "$SRC"

    sed -i "$expr" "$SRC"
    if diff -q "$BACKUP" "$SRC" >/dev/null 2>&1; then
        echo "MUT-NOOP     $name  (sed changed nothing — the pattern is stale)"
        NOOP=$((NOOP + 1))
        cp "$BACKUP" "$SRC"
        return
    fi

    # Force a rebuild of the mutated translation unit (the harness never leaves
    # a mutated .o cached in the real build dir).
    rm -f "build/utils/temperature_scaling.o" "build/utils/temperature_scaling.d"

    local out
    out="$(make "$BIN" 2>&1)"
    if ! printf '%s' "$out" | grep -q "error:"; then
        if [ ! -f "$BIN" ]; then
            echo "MUT-BROKEN   $name  (no binary produced)"
            BROKEN=$((BROKEN + 1))
            cp "$BACKUP" "$SRC"
            return
        fi
        ./"$BIN" >"$TMPBIN.out" 2>&1
        local rc=$?
        local n
        n=$(grep -c "\[FAIL\]" "$TMPBIN.out" 2>/dev/null || echo 0)
        if [ "$rc" -ne 0 ]; then
            local verb="MUT-CAUGHT"
            if [ "$rc" -gt 128 ]; then
                verb="MUT-CAUGHT(signal $((rc - 128)))"
            fi
            echo "$verb   $name  (by $n assertions, exit $rc)"
            CAUGHT=$((CAUGHT + 1))
        else
            echo "MUT-SURVIVED $name  <-- investigate the test, not the code"
            SURVIVED=$((SURVIVED + 1))
            FAILED_MUTATIONS+=("$name")
        fi
    else
        echo "MUT-BROKEN   $name  (compile error — proves nothing about test strength)"
        BROKEN=$((BROKEN + 1))
    fi
    rm -f "$TMPBIN.out"
    cp "$BACKUP" "$SRC"
}

echo
echo "=== Mutations ==="

# run_mutation_expect_invert <name> <sed-expr> <probe-cpp>
#
# The TDD skill's four vacuity modes include "a mutation that provably changes
# nothing is not a vacuous test" — and the way to tell an inert mutation from a
# real test gap is to MEASURE, then invert the verdict so that a WRONG inert
# claim fails. This function asserts that the suite PASSES with the mutation
# applied AND that a probe's outputs are bit-identical with and without it.
# If either changes, the harness fails — so "inert" can never be asserted by
# assumption.
run_mutation_expect_inert() {
    local name="$1"; shift
    local expr="$1"; shift
    local probe="$1"; shift

    # 1. Output WITH the mutation.
    cp "$BACKUP" "$SRC"
    sed -i "$expr" "$SRC"
    if diff -q "$BACKUP" "$SRC" >/dev/null 2>&1; then
        echo "INERT-NOOP  $name  (sed changed nothing — cannot be classified)"
        NOOP=$((NOOP + 1))
        cp "$BACKUP" "$SRC"
        return 1
    fi
    rm -f "build/utils/temperature_scaling.o" "build/utils/temperature_scaling.d"
    make "$BIN" >/dev/null 2>&1
    local mut_bin_ok=1
    if ./"$BIN" >/dev/null 2>&1; then
        :  # suite passes with the mutation, as an inert mutation requires
    else
        mut_bin_ok=0
    fi
    g++ -std=c++17 -O2 -Iinclude -o "$TMPBIN" "$probe" "$SRC" \
        include/nn/utils/losses.cpp include/nn/core/tensor.cpp >/dev/null 2>&1
    "$TMPBIN" >"$TMPBIN.mut" 2>&1
    cp "$BACKUP" "$SRC"

    # 2. Output WITHOUT the mutation (pristine).
    rm -f "build/utils/temperature_scaling.o" "build/utils/temperature_scaling.d"
    make "$BIN" >/dev/null 2>&1
    g++ -std=c++17 -O2 -Iinclude -o "$TMPBIN" "$probe" "$BACKUP" \
        include/nn/utils/losses.cpp include/nn/core/tensor.cpp >/dev/null 2>&1
    "$TMPBIN" >"$TMPBIN.base" 2>&1

    # 3. INVERTED VERDICT: inert means IDENTICAL. Any difference fails.
    if [ "$mut_bin_ok" -eq 0 ]; then
        echo "INERT-FAIL  $name  (the suite FAILS with the mutation, so it is NOT inert)"
        SURVIVED=$((SURVIVED + 1))
        FAILED_MUTATIONS+=("$name (claimed inert but caught)")
        rm -f "$TMPBIN.mut" "$TMPBIN.base"
        return 1
    fi
    if ! diff -q "$TMPBIN.base" "$TMPBIN.mut" >/dev/null 2>&1; then
        echo "INERT-FAIL  $name  (probe output DIFFERS — the mutation is NOT inert)"
        diff "$TMPBIN.base" "$TMPBIN.mut" | head -6
        SURVIVED=$((SURVIVED + 1))
        FAILED_MUTATIONS+=("$name (claimed inert but outputs differ)")
        rm -f "$TMPBIN.mut" "$TMPBIN.base"
        return 1
    fi
    echo "MUT-INERT   $name  (suite passes AND probe output is bit-identical)"
    rm -f "$TMPBIN.mut" "$TMPBIN.base"
    return 0
}

# M1: divide -> multiply. The single most important mutation; the paper's Eq. 9
# is a division and an inverted op produces plausible-looking probabilities.
run_mutation "M1: /T -> *T" 's|scaled(i, j) = logits(i, j) / T;|scaled(i, j) = logits(i, j) * T;|'

# M2: drop the 1/(n T^2) factor from the gradient.
run_mutation "M2: drop the 1/T^2 factor in dL/dT" \
    's|const double inv_T2 = 1.0 / (T \* T);|const double inv_T2 = 1.0;|'

# M3: reverse the gradient bracket. The sign fingerprint is rel = 2.0 vs FD.
run_mutation "M3: reverse the dL/dT bracket" \
    's|total += (logits(i, y) - pz) \* inv_T2;|total += (pz - logits(i, y)) * inv_T2;|'

# M4: drop the softmax-weighted logit term sum_j p_j z_j.
run_mutation "M4: drop the sum_j p_j z_j term" \
    's|double pz = 0.0;\n|X|; s|for (size_t j = 0; j < logits.cols; ++j) pz += probs(i, j) \* logits(i, j);|for (size_t j = 0; j < logits.cols; ++j) pz += probs(i, j);|'

# M5: degenerate the fit to the identity (never search).
run_mutation "M5: fit always returns T = 1" \
    's|double lo = std::log(t_min);|if (true) return 1.0;\n    double lo = std::log(t_min);|'

# M6: sum instead of mean in the NLL reduction (must target the NLL function,
# line-anchored, since there are two identical reduction lines in this file).
run_mutation "M6: NLL sum instead of mean" \
    's|    // MEAN over rows, per the header.s note (5). A sum would be a different|    if (true) return total;\n    // MEAN over rows, per the header.s note (5). A sum would be a different|'

# M7: ignore the bracket clamp on the way out. The replacement must be a
# SYNTAX-VALID no-op statement, not a bare `X`: deleting the line with `X`
# leaves a dangling `X` and the mutant fails to COMPILE, which the harness
# reports as MUT-BROKEN — correct behaviour, but it then proves nothing about
# the tests. A mutation that cannot build is not a test-strength experiment.
#
# M7 is EXPECTED TO BE INERT, and that is a claim that must be MEASURED rather
# than assumed (see run_mutation_expect_inert). best_T is initialised to t_min
# and thereafter only ever assigned t_max or exp(mid) for mid in [lo, hi], so it
# cannot leave [t_min, t_max] and the clamp never fires. The probe below covers
# all four assignment paths, and the harness fails if ANY output differs.
run_mutation_expect_inert "M7: drop the result clamp to [t_min, t_max]" \
    's|    if (best_T < t_min) best_T = t_min;|    (void)0;|' \
    "tests/probe_inert_m7.cpp"

# M8: endpoints never evaluated as candidates (this is the bug that made a
# separable set return 5.8e-2 instead of t_min).
run_mutation "M8: never evaluate the bracket endpoints" \
    's|    double best_T = t_min;|    double best_T = std::exp((lo + hi) / 2.0);|; s|    const double upper_L = temperature_nll(logits, labels, t_max);|    const double upper_L = best_L;|'

echo
echo "=== Summary ==="
echo "  MUT-CAUGHT:   $CAUGHT"
echo "  MUT-SURVIVED: $SURVIVED"
echo "  MUT-BROKEN:   $BROKEN"
echo "  MUT-NOOP:     $NOOP"
echo "  (M7 is classified MUT-INERT above, via a differential probe whose"
echo "   verdict is INVERTED — a wrong 'inert' claim fails the harness.)"
if [ "$SURVIVED" -gt 0 ]; then
    echo
    echo "Survivors (a survivor is a QUESTION ABOUT THE TEST):"
    for m in "${FAILED_MUTATIONS[@]}"; do echo "  - $m"; done
fi
cp "$BACKUP" "$SRC"
[ "$SURVIVED" -eq 0 ] && [ "$NOOP" -eq 0 ] && [ "$BROKEN" -eq 0 ]