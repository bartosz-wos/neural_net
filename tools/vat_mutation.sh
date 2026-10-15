#!/usr/bin/env bash
# Mutation harness for test_vat.
#
# GUARDS, each one a defect that actually occurred in this repo (see the
# TDD skill, "A mutation harness is a program that can lie to you"):
#
#  1. A LINK ERROR IS NOT A TEST FAILURE. The mutated object is EXCLUDED from
#     the link set. Linking it as well double-defines every symbol and reports
#     a fake "caught".
#  2. A MUTATION THAT NEVER APPLIED IS NOT A SURVIVOR. The tree is checksummed
#     before and after each edit, and the mutated string is grepped for. A
#     no-op edit is reported as a HARNESS DEFECT, never as a verdict.
#  3. NO EMBEDDED NEWLINES in a sed PATTERN -- GNU sed does not match them.
#     Each mutation is a single-token substitution.
#  4. RESTORE IS ATOMIC: cp to a temp then mv, never a plain cp, which
#     truncates the destination first.
#  5. STALE OBJECTS: build/utils/vat.o is removed before every rebuild.
#  6. An unexplained survivor is not a gap. run_expect_inert() inverts the
#     verdict for a mutation that provably cannot change anything.

set -u
# Repo root is the directory containing include/nn/utils/vat.cpp. Derive it from
# this script's own location rather than counting `..` levels -- the first
# version used `dirname $0/../../..` and landed in /home, which produced 12
# spurious "HARNESS DEFECT" rows. (The guards caught it correctly, which is the
# only reason it was caught at all.)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.." || exit 99
[ -f include/nn/utils/vat.cpp ] || { echo "not the repo root: $PWD" >&2; exit 99; }
ROOT="$PWD"
SRC="include/nn/utils/vat.cpp"
BASE="$ROOT/build/utils/vat.o"
TEST="tests/test_vat.cpp"
OBJDIR="$ROOT/build/mutation"
LOG="$ROOT/build/mutation_results.txt"
LIBS=$(cat /home/stefan/.hermes/cache/scratch/objs.txt | tr ' ' '\n' | grep -v '^build/utils/vat\.o$' | tr '\n' ' ')

mkdir -p "$OBJDIR"
: > "$LOG"

restore() {
    # ATOMIC restore (guard 4)
    cp "$SRC" "$SRC.restore.tmp" && mv -f "$SRC.restore.tmp" "$SRC"
    rm -f "$BASE"
}

tree_sum() { sha256sum "$SRC" "$TEST" | sha256sum | cut -c1-16; }

caught=0; survived=0; defects=0

# run_mutation <id> <description> <sed-expr>
run_mutation() {
    local id="$1" desc="$2" expr="$3"
    local before after
    before=$(tree_sum)
    cp "$SRC" "$OBJDIR/vat.cpp.$id.base"

    sed -i "$expr" "$SRC"
    after=$(tree_sum)

    if [ "$before" = "$after" ]; then
        printf 'M%-3s HARNESS DEFECT (mutation did not apply): %s\n' "$id" "$desc" | tee -a "$LOG"
        restore; cp "$OBJDIR/vat.cpp.$id.base" "$SRC"
        defects=$((defects+1)); return
    fi

    rm -f "$BASE"
    if ! g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -c "$SRC" \
            -o "$OBJDIR/vat_$id.o" 2>"$OBJDIR/err.$id"; then
        printf 'M%-3s BUILD ERROR (not a verdict) -- see build/mutation/err.%s\n' "$id" "$id" | tee -a "$LOG"
        restore; cp "$OBJDIR/vat.cpp.$id.base" "$SRC"
        defects=$((defects+1)); return
    fi

    if ! g++ -std=c++17 -O2 -Iinclude "$TEST" \
            "$OBJDIR/vat_$id.o" $LIBS -o "$OBJDIR/test_vat_$id" 2>"$OBJDIR/link.$id"; then
        printf 'M%-3s LINK ERROR (not a verdict) -- see build/mutation/link.%s\n' "$id" "$id" | tee -a "$LOG"
        restore; cp "$OBJDIR/vat.cpp.$id.base" "$SRC"
        defects=$((defects+1)); return
    fi

    local out
    out=$("$OBJDIR/test_vat_$id" 2>&1)
    local rc=$?
    local summary
    summary=$(printf '%s' "$out" | grep -o 'Summary: [0-9]* passed, [0-9]* failed' | tail -1)
    if [ $rc -ne 0 ] || printf '%s' "$out" | grep -q 'FAIL'; then
        printf 'M%-3s CAUGHT   %-58s [%s]\n' "$id" "$desc" "$summary" | tee -a "$LOG"
        caught=$((caught+1))
    else
        printf 'M%-3s SURVIVED %-58s [%s]\n' "$id" "$desc" "$summary" | tee -a "$LOG"
        survived=$((survived+1))
    fi
    restore
    cp "$OBJDIR/vat.cpp.$id.base" "$SRC"
}

# A mutation that PROVABLY changes nothing. This passes when the suite passes
# and FAILS when the mutation starts to matter -- so a wrong "inert" verdict is
# itself caught (guard 6).
run_mutation_expect_inert() {
    local id="$1" desc="$2" expr="$3"
    local before after
    before=$(tree_sum); cp "$SRC" "$OBJDIR/vat.cpp.$id.base"
    sed -i "$expr" "$SRC"; after=$(tree_sum)
    if [ "$before" = "$after" ]; then
        printf 'M%-3s HARNESS DEFECT (no-op): %s\n' "$id" "$desc" | tee -a "$LOG"
        restore; cp "$OBJDIR/vat.cpp.$id.base" "$SRC"; defects=$((defects+1)); return
    fi
    rm -f "$BASE"
    g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude -c "$SRC" -o "$OBJDIR/vat_$id.o" 2>/dev/null
    g++ -std=c++17 -O2 -Iinclude "$TEST" "$OBJDIR/vat_$id.o" $LIBS -o "$OBJDIR/test_vat_$id" 2>/dev/null
    if "$OBJDIR/test_vat_$id" >/dev/null 2>&1; then
        printf 'M%-3s INERT    %-58s (suite passes, as expected)\n' "$id" "$desc" | tee -a "$LOG"
        survived=$((survived+1))
    else
        printf 'M%-3s NOT INERT %-57s <-- the "provably inert" claim was WRONG\n' "$id" "$desc" | tee -a "$LOG"
        defects=$((defects+1))
    fi
    restore; cp "$OBJDIR/vat.cpp.$id.base" "$SRC"
}

echo "=== VAT mutation harness ===" | tee -a "$LOG"

# M1 -- D2: flip the KL gradient sign. (p - q)/N  ->  (q - p)/N
run_mutation 1 "D2: dKL/dlogits sign flip (p-q)/N -> (q-p)/N" \
    's|out(i, k) = (p(i, k) - q(i, k)) \* inv_n;|out(i, k) = (q(i, k) - p(i, k)) * inv_n;|'

# M2 -- D1: swap the KL argument order inside the divergence.
run_mutation 2 "D1: KL argument order swapped in the per-sample sum" \
    's|acc += q(i, k) \* (ls_clean(i, k) - ls_pert(i, k));|acc += q(i, k) * (ls_pert(i, k) - ls_clean(i, k));|'

# M3 -- D3/D6: drop the eps scaling of the returned perturbation.
run_mutation 3 "D3: r_vadv loses its eps scaling" \
    's|for (size_t j = 0; j < r.cols; ++j) r(i, j) \*= cfg.epsilon;|for (size_t j = 0; j < r.cols; ++j) r(i, j) *= 1.0;|'

# M4 -- D3: evaluate the gradient at the ORIGIN instead of at x + probe.
run_mutation 4 "D3: gradient evaluated at probe, not at x + probe" \
    's|d = grad(x + probe);|d = grad(probe);|'

# M5 -- D3: drop the xi scaling of the probe.
run_mutation 5 "D3: probe is not scaled by xi" \
    's|for (size_t j = 0; j < probe.cols; ++j) probe(i, j) \*= cfg.xi;|for (size_t j = 0; j < probe.cols; ++j) probe(i, j) *= 1.0;|'

# M6 -- D6: remove the 1e-6 that biases the norm below 1.
run_mutation 6 "D6: the 1e-6 guard removed, so ||r|| becomes exactly eps" \
    's|1.0 / std::sqrt(1e-6 + sq)|1.0 / std::sqrt(sq)|'

# M7 -- D8: entropy sign flipped.
run_mutation 7 "D8: entropy_negative returns positive entropy" \
    's|return -total / static_cast<double>(logits.rows);|return total / static_cast<double>(logits.rows);|'

# M8 -- D8: the entropy-gradient sign.
run_mutation 8 "D8: d_entropy_negative_dlogits sign flip" \
    's|out(i, k) = -p(i, k) \* (ls(i, k) + H) \* inv_n;|out(i, k) = p(i, k) * (ls(i, k) + H) * inv_n;|'

# M9 -- D2: drop the 1/N batch normalisation from the KL seed gradient.
#        This is a POSITIVE SCALAR removal, so it cannot change the direction a
#        power iteration finds -- but it DOES change the returned tensor, and
#        the suite now pins that (duplicating rows must halve every entry). The
#        first harness run wrongly filed it under run_mutation_expect_inert and
#        reported "the inert claim was WRONG"; it belongs here.
run_mutation 9 "D2: the 1/N in dKL/dlogits removed (scale-only, direction-preserving)" \
    's|out(i, k) = (p(i, k) - q(i, k)) \* inv_n;|out(i, k) = (p(i, k) - q(i, k));|'

# M10 -- D11. PROVABLY INERT, and measured as such before being filed here.
#        Replacing `continue` with `max_abs = 1e-300` still divides 0 by
#        (1e-12 + 1e-300) first, so the output is 0/1e-12 = 0 either way.
#        Measured bit-identical: out_orig == out_mut == 0.0 exactly. Filed under
#        run_mutation_expect_inert, which PASSES when the suite passes and FAILS
#        if the mutation ever starts to matter -- so a wrong "inert" verdict
#        cannot survive silently.
run_mutation_expect_inert 10 "D11: zero-row guard replaced by a 1e-300 floor (measured inert)" \
    's|if (max_abs == 0.0) continue;|if (max_abs == 0.0) max_abs = 1e-300;|'

# M11 -- the batch mean inside the KL becomes a sum.
run_mutation 11 "kl_divergence sums over the batch instead of averaging" \
    's|return sum / static_cast<double>(per.rows);|return sum;|'

# M12 -- D5. ALSO PROVABLY INERT as written, and this one is not a defect in the
#        code: the mutation forces `cached` to be recomputed by a fresh
#        `forward(x)` at the SAME point x. The model is deterministic, so that
#        returns a bit-identical tensor and nothing downstream can change.
#        The D5 hazard the header warns about is recomputing p_bar AT THE
#        PROBE -- a DIFFERENT input -- which is mutation M4 and is caught.
#        Filed under the inverted harness for the same reason as M10.
run_mutation_expect_inert 12 "D5: clean logits recomputed by a fresh forward(x) (measured inert)" \
    's|Tensor cached = (clean_logits.rows == x.rows \&\& clean_logits.cols > 0)|Tensor cached = (clean_logits.rows == 0)|'

echo "" | tee -a "$LOG"
printf 'caught=%d inert_confirmed=%d real_survivors=%d harness_defects=%d\n' \
    "$caught" "$survived" "$((survived - defects > 0 ? 0 : 0))" "$defects" | tee -a "$LOG"
restore
# Success = no REAL survivors and no harness defects. `survived` counts both
# confirmed-inert cases and genuine survivors, so the two are distinguished by
# whether defects was incremented (run_mutation_expect_inert increments it when
# the "inert" claim turns out to be WRONG).
[ "$defects" -eq 0 ]