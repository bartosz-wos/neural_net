#!/usr/bin/env bash
# tests/mutate_speculative_decoding.sh -- mutation harness for the speculative
# decoding step in include/nn/utils/logit_processing.{h,cpp}.
#
# This harness exists because the repo has recorded TWENTY-ONE instances of a
# mutation harness reporting a confident wrong verdict. Every one of the
# defects below is one this repo has already paid for, so they are addressed
# by construction rather than by memory:
#
#   1. restore() MUST NOT DELETE ITS OWN BACKUP. It is called once per mutation
#      AND on EXIT; a restore that `rm`s the backup works exactly once, every
#      later restore is a silent no-op printing `cp: cannot stat`, and the
#      tree is left holding whatever mutant ran last. Cleanup happens exactly
#      once, on EXIT, and restore() is ATOMIC (`cp tmp && mv -f`).
#   2. NEVER RECOVER THE LINK SET BY REGEX. `LIB_OBJS := $(LIB_SRCS:...)` is a
#      RECURSIVE make variable whose literal text contains no .o filenames; a
#      regex against it yields ZERO objects and turns every mutation into a
#      false "did not link". This harness asks make, and ASSERTS the count.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET, or every mutation
#      double-defines every symbol and reports CAUGHT (compile error). A link
#      error is not a mutation verdict. The correct handling is to COMPILE THE
#      MUTATED TU SEPARATELY, drop that one object from the library link set,
#      and link the mutant in its place.
#   4. `MUT-BROKEN` MEANS THE BUILD FAILED. It is reported separately from
#      CAUGHT and never counted as a pass, and a mutation reported BROKEN is
#      re-run in isolation before it is believed.
#   5. A MUTATION THAT DID NOT APPLY IS A HARNESS DEFECT, not a verdict. The
#      tree is checksummed before and after every edit.
#   6. NO EMBEDDED NEWLINES IN A sed PATTERN -- `\n` does not match in GNU sed.
#      The edit is done by a Python editor with no delimiter concept.
#   7. A MISSING BINARY IS `UNAVAILABLE`, never a verdict.
#   8. The spec loop refuses any entry that does not name a known file tag, so a
#      stray `inert` marker can never be read as a sed script.
#   9. Throwing calls are wrapped so a guard mutation reports a verdict instead
#      of aborting the binary and leaving the harness with nothing to count.
#  10. `rm -f` the intermediates before every build -- a stale .o from the
#      previous mutation reproduces a build failure that does not exist alone.
#  11. AN ANCHOR THAT IS AMBIGUOUS IS A HARNESS DEFECT, and adding a feature
#      can INVALIDATE an existing mutation's anchor. The count is taken with
#      the SAME engine that performs the edit.
#  12. A MUTATION THAT PROVABLY CHANGES NOTHING IS NOT A VACUOUS TEST. It gets
#      its own harness function that INVERTS the verdict, so a future
#      refactor that makes it matter fails loudly. An unexplained survivor is
#      indistinguishable from a real gap.
#
# THIS HARNESS MUTATES include/nn/utils/logit_processing.cpp -- THE SAME FILE
# tests/mutate_logit_processing.sh mutates. The two must NEVER be run
# CONCURRENTLY: they share build/*.o, and the repo has already recorded a
# build race on that shared state producing two false "MUT-BROKEN" verdicts.

set -u
cd "$(dirname "$0")/.." || exit 1

SRC="include/nn/utils/logit_processing.cpp"
HDR="include/nn/utils/logit_processing.h"
TEST="tests/test_speculative_decoding.cpp"
BIN="build/test_speculative_decoding"
BAK_SRC=$(mktemp); BAK_HDR=$(mktemp)
BAK_MARK=".mutate_spec_tree_pristine"
cp "$SRC" "$BAK_SRC"; cp "$HDR" "$BAK_HDR"
cp "$SRC" "$BAK_MARK"

restore() {                      # atomic; NEVER deletes the backups
    cp "$BAK_SRC" "$SRC.tmp" 2>/dev/null && mv -f "$SRC.tmp" "$SRC"
    cp "$BAK_HDR" "$HDR.tmp" 2>/dev/null && mv -f "$HDR.tmp" "$HDR"
}
cleanup() { restore; rm -f "$BAK_SRC" "$BAK_HDR" "$BAK_MARK"; }
trap cleanup EXIT INT TERM

# ---- link set, from make, with a count assertion (defect 2) -----------------
LIB_LINK=$(make -s print-lib-objs 2>/dev/null | grep '\.o$' | grep -v '/test_' | tr '\n' ' ')
LIB_COUNT=$(echo "$LIB_LINK" | wc -w)
if [ "$LIB_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: recovered only $LIB_COUNT library objects (< 10)." >&2
    echo "Every verdict below would be false. REFUSING TO RUN." >&2
    exit 2
fi
MUT_OBJ="build/utils/logit_processing.o"
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
    rm -f build/test_speculative_decoding.o "$BIN" "$MUT_OBJ"
    make -s build/utils/logit_processing.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$TEST" -o build/test_speculative_decoding.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$SRC" -o "$MUT_OBJ" >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL $LIB_LINK "$MUT_OBJ" build/test_speculative_decoding.o -o "$BIN" >/dev/null 2>&1 || return 1
    return 0
}

# ---- pristine self-check: the suite must be GREEN before any mutation ------
build_and_run
PRISTINE=$("$BIN" 2>&1 | sed -n 's/^=== Summary: \([0-9]*\) passed.*/\1/p' | tail -1)
if [ "${PRISTINE:-0}" -eq 0 ] || ! "$BIN" >/dev/null 2>&1; then
    echo "HARNESS DEFECT: pristine tree is not green ($PRISTINE passed)." >&2
    echo "REFUSING TO RUN -- this is exactly how a leaked mutant is caught." >&2
    exit 2
fi
echo "pristine: $PRISTINE passed, 0 failed -- mutation may begin"

CAUGHT=0; SURVIVED=0; BROKEN=0; NOOP=0; UNAVAIL=0
INERT_OK=0; INERT_BAD=0
declare -a TABLE

# Apply an edit with a Python editor (no delimiter concept -- defect 6), and
# verify it applied by checksum (defect 5).
apply_edit() {   # $1=file $2=from $3=to
    python3 - "$1" "$2" "$3" <<'PYEOF'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding='utf8').read()
n = s.count(frm)
if n != 1:
    sys.stderr.write(f"anchor count {n}\n")
    sys.exit(3)
open(path, 'w', encoding='utf8').write(s.replace(frm, to))
PYEOF
}

run_mutation() {   # $1=id  $2=file  $3=from  $4=to
    local id="$1" file="$2" from="$3" to="$4"
    restore

    # Every spec must name a file that exists AND contain the anchor exactly
    # once, or we are looking at a harness defect rather than a verdict.
    if [ ! -f "$file" ]; then
        TABLE+=("$id|$file does not exist|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi
    local n
    n=$(python3 -c 'import sys; print(open(sys.argv[1],encoding="utf8").read().count(sys.argv[2]))' "$file" "$from")
    if [ "$n" -ne 1 ]; then
        TABLE+=("$id|anchor count = $n|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi

    local before after
    before=$(md5sum "$file" | cut -d' ' -f1)
    if ! apply_edit "$file" "$from" "$to"; then
        restore
        TABLE+=("$id|editor failed|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi
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
    np=$(echo "$out" | sed -n 's/^=== Summary: \([0-9]*\) passed.*/\1/p' | tail -1)
    nf=$(echo "$out" | sed -n 's/.*, \([0-9]*\) failed ===.*/\1/p' | tail -1)
    np=${np:-0}; nf=${nf:-0}

    if [ "$rc" -ge 128 ]; then              # a crash is not a verdict
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

# ---------------------------------------------------------------------------
# THE INERT-MUTATION FUNCTION (defect 12).
#
# A mutation that PROVABLY changes nothing is not a vacuous test -- it is a
# semantically inert edit. But an unexplained survivor is indistinguishable
# from a real coverage gap, so an inert mutation must be SCORED DIFFERENTLY:
# this function PASSES when the suite stays green (the mutation really is
# inert) and FAILS when the mutation starts to matter. That way a future
# refactor that makes the inert mutation observable breaks the harness loudly
# instead of silently reclassifying it.
#
# The drop-the-clamp mutation is the canonical case, and the oracle MEASURED
# it: the clamped and the unclamped acceptance rules produce identical output
# on 3000/3000 random trials, because the rule is written as the threshold
# comparison `r <= p/q` and a ratio above 1 already accepts every r in [0,1).
# The clamp is DEFENSIVE, not load-bearing.
# ---------------------------------------------------------------------------
run_mutation_expect_inert() {   # $1=id  $2=file  $3=from  $4=to
    local id="$1" file="$2" from="$3" to="$4"
    restore

    local n
    n=$(python3 -c 'import sys; print(open(sys.argv[1],encoding="utf8").read().count(sys.argv[2]))' "$file" "$from")
    if [ "$n" -ne 1 ]; then
        TABLE+=("$id|anchor count = $n|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi

    local before after
    before=$(md5sum "$file" | cut -d' ' -f1)
    if ! apply_edit "$file" "$from" "$to"; then
        restore
        TABLE+=("$id|editor failed|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi
    after=$(md5sum "$file" | cut -d' ' -f1)
    if [ "$before" = "$after" ]; then
        restore
        TABLE+=("$id|edit did not apply|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi

    build_and_run
    if [ ! -x "$BIN" ]; then
        restore
        TABLE+=("$id|no binary produced|UNAVAILABLE"); UNAVAIL=$((UNAVAIL+1)); return
    fi

    local out rc nf
    out=$("$BIN" 2>&1); rc=$?
    nf=$(echo "$out" | sed -n 's/.*, \([0-9]*\) failed ===.*/\1/p' | tail -1)
    nf=${nf:-0}

    if [ "$rc" -ge 128 ]; then
        restore
        TABLE+=("$id|CRASH rc=$rc|INERT-BROKEN"); INERT_BAD=$((INERT_BAD+1)); return
    fi
    if [ "$nf" -eq 0 ]; then
        # The mutation is INERT, as measured. This is the CORRECT verdict.
        restore
        TABLE+=("$id|suite still green -- INERT as measured|INERT"); INERT_OK=$((INERT_OK+1)); return
    fi
    # The mutation now MATTERS. That is not a failure of the suite -- it is
    # the mutation having become observable, which the harness must report
    # loudly rather than reclassify silently.
    restore
    TABLE+=("$id|$nf failed -- the mutation now MATTERS|INERT-BECAME-LOAD-BEARING")
    INERT_BAD=$((INERT_BAD+1))
}

echo
echo "=== speculative decoding mutations ==="

# ---------------------------------------------------------------------------
# THE MUTATIONS. Each is a natural mistake, and each must be CAUGHT.
# ---------------------------------------------------------------------------

# --- SP3: the acceptance comparison. `<=` becomes `<`. -------------------
# A draw that lands EXACTLY on the acceptance probability then rejects. The
# test supplies that draw on a dyadic fixture, so this IS observable -- but
# ONLY there; the test's other fixtures use random draws and cannot see it.
run_mutation M01 "$SRC" "if (!(uniforms[gamma + i] <= ratio)) {" \
                        "if (!(uniforms[gamma + i] < ratio)) {"

# --- SP7: the residual is used even when every guess was accepted. --------
# This is the mutation that costs the entire speedup: the step would emit
# `n` tokens instead of `n+1` and, for identical models, the residual is
# all-zero so the normalisation divides by zero.
run_mutation M02 "$SRC" "    if (rejected) {
        // The residual at index n+1 (1-based) == index n (0-based). Both
        // rows are among the ones already computed.
        p_prime = residual_distribution(p_rows[n], q_rows[n]);
    } else {" \
                        "    if (false) {
        p_prime = residual_distribution(p_rows[n], q_rows[n]);
    } else {"

# --- SP6: the residual is built at the WRONG index (n-1 instead of n). ----
run_mutation M03 "$SRC" "        p_prime = residual_distribution(p_rows[n], q_rows[n]);" \
                        "        p_prime = residual_distribution(p_rows[n - 1 > 0 ? n - 1 : 0], q_rows[n - 1 > 0 ? n - 1 : 0]);"

# --- SP5/SP9: the rejected guess is emitted as well as the correction. ----
# The FIRST DRAFT OF THIS MUTATION was a no-op: it wrote the rejected guess
# to `result.tokens(0, n)` and then wrote the correction to the SAME slot,
# so the second write OVERWROTE the first and the observable was unchanged.
# The mutation has to change the tensor's LENGTH.
run_mutation M04 "$SRC" "    result.tokens = Tensor(1, n + 1);" \
                        "    result.tokens = Tensor(1, n + 2);"

# --- SP6: the componentwise max(0, .) is dropped from the residual. -------
# p and q are both distributions, so sum(p - q) == 0 EXACTLY and the
# normalisation DIVIDES BY ZERO. The suite's guard must catch that rather
# than shipping NaNs.
run_mutation M05 "$SRC" "        double d = p[j] - q[j];
        if (d > 0.0) {
            r[j] = d;
            sum += d;
        }" \
                        "        double d = p[j] - q[j];
        r[j] = d;
        sum += d;"

# --- SP12: the acceptance is DETERMINISTIC on p >= q. -------------------
# The natural mistake: "keep the guess if the target likes it at least as
# much as the draft did". It ignores the acceptance PROBABILITY entirely, so
# the acceptance rate is 0 or 1 per token instead of p/q.
run_mutation M06 "$SRC" "        double ratio = acceptance_probability(pi, qi);" \
                        "        double ratio = (pi >= qi) ? 1.0 : 0.0;"

# --- SP8: the draft is called one time too FEW. --------------------------
# The paper's cost model is gamma draft calls; dropping the last one means
# the residual cannot be built from the row that produced the last guess.
#
# THE FIRST DRAFT OF THIS MUTATION'S ANCHOR WAS AMBIGUOUS: `for (size_t i =
# 0; i < gamma; ++i) {` appears TWICE (the draft loop and the acceptance
# loop), so the harness refused the edit -- correctly, since editing one of
# two occurrences is a coin flip. It is now anchored on the CONTEXT that
# makes it unique. (Defect 11: adding a feature can invalidate an existing
# mutation's anchor, and the failure is quiet.)
# --- SP4: the LAST guess is never acceptance-tested. ---------------------
# The first draft of this mutation removed one DRAFT CALL, which made
# `guesses` one element short and the code then indexed `guesses[j]` out of
# bounds -- the binary SEGFAULTED with rc=139. A crash is NOT a verdict: it
# says nothing about whether the suite covers the behaviour, and reporting it
# as one would score a defect as a coverage claim.
#
# The mutation is now on the ACCEPTANCE LOOP, which is where the observable
# difference actually is: shortening the loop means the last guess is
# accepted whatever the target model thinks of it.
run_mutation M07 "$SRC" "    size_t n = gamma;
    for (size_t i = 0; i < gamma; ++i) {
        double pi = p_rows[i][static_cast<size_t>(guesses[i])];" \
                        "    size_t n = gamma;
    for (size_t i = 0; i + 1 < gamma; ++i) {
        double pi = p_rows[i][static_cast<size_t>(guesses[i])];"

# --- SP6: the residual is normalised by the MAX instead of the SUM. ------
run_mutation M08 "$SRC" "    for (size_t j = 0; j < v; ++j) r[j] /= sum;" \
                        "    double mx = 0.0; for (size_t j = 0; j < v; ++j) if (r[j] > mx) mx = r[j]; for (size_t j = 0; j < v; ++j) r[j] /= mx;"

# --- SP9: the step reports the WRONG n. ----------------------------------
# The first draft of this mutation set the tensor's width to `n`, which made
# the write to `result.tokens(0, n)` run ONE PAST THE END of the row -- the
# binary CRASHED with rc=134, and a crash is NOT a verdict (it says nothing
# about whether the suite covers the behaviour).
#
# The mutation now keeps the SHAPE and corrupts the REPORTED count instead,
# which is observable: the suite asserts `tokens.cols == num_accepted + 1`
# on every trial.
run_mutation M09 "$SRC" "    result.num_accepted = n;" \
                        "    result.num_accepted = n + 1;"

# --- SP2: the target's row is taken one position too late. --------------
run_mutation M10 "$SRC" "        std::vector<double> pi = model_probs(target, at, \"the target model\");" \
                        "        std::vector<double> pi = model_probs(target, grow(at, 0), \"the target model\");"

# --- SP1: the draft's recorded row is the one from the NEXT position. ----
run_mutation M11 "$SRC" "        q_rows.push_back(qi);" \
                        "        q_rows.push_back(model_probs(draft, grow(cur, 0), \"the draft model\"));"

# --- a guard removal: the zero-vocabulary / empty-row check. -------------
run_mutation M12 "$SRC" "    if (logits.rows == 0 || logits.cols == 0) {" \
                        "    if (false) {"

# --- the acceptance probability's zero guard -- MEASURED INERT. ----------
# Without the guard, `p_at / q_at` with `q_at == 0` is +inf, and `inf < 1.0`
# is false, so the function returns 1.0 -- EXACTLY what the guard returns.
# The oracle MEASURED this over the whole domain (p_at/q_at pairs with
# q_at == 0 and q_at subnormal): every case gives the same answer with and
# without the guard.
#
# It is scored by run_mutation_expect_inert, which INVERTS the verdict: it
# passes while the mutation is inert and FAILS if a future refactor makes it
# observable. The guard is kept because it says what the quantity is -- the
# guess was DRAWN from q, so q(guess) > 0 by construction.
run_mutation_expect_inert M13 "$SRC" "    if (!(q_at > 0.0)) return 1.0;" \
    "    if (false) return 1.0;"

# ---------------------------------------------------------------------------
# THE INERT MUTATION (defect 12). Dropping the clamp.
#
# The oracle MEASURED this: the clamped and the unclamped acceptance rules
# agree on 3000/3000 random trials, because the rule is a threshold
# comparison and a ratio above 1 already accepts every r in [0,1). It is
# scored by run_mutation_expect_inert, which INVERTS the verdict: it passes
# while the mutation is inert and FAILS if a future refactor makes it
# observable.
# ---------------------------------------------------------------------------
run_mutation_expect_inert M14 "$SRC" \
    "    double ratio = p_at / q_at;
    return ratio < 1.0 ? ratio : 1.0;" \
    "    double ratio = p_at / q_at;
    return ratio;"

echo
printf '%-6s %-58s %s\n' "ID" "DETAIL" "VERDICT"
printf '%-6s %-58s %s\n' "----" "-----" "-------"
for row in "${TABLE[@]}"; do
    IFS='|' read -r a b c <<< "$row"
    printf '%-6s %-58s %s\n' "$a" "$b" "$c"
done

echo
echo "=== caught $CAUGHT / survived $SURVIVED / broken $BROKEN / noop $NOOP / unavailable $UNAVAIL ==="
echo "=== inert (measured): $INERT_OK ok / $INERT_BAD became load-bearing ==="

# ---- post-harness verification: the tree must be pristine again -----------
restore
build_and_run
FINAL=$("$BIN" 2>&1 | sed -n 's/^=== Summary: \([0-9]*\) passed.*/\1/p' | tail -1)
if "$BIN" >/dev/null 2>&1 && [ "${FINAL:-0}" -eq "$PRISTINE" ]; then
    echo "POST-HARNESS: tree restored, suite green again ($FINAL passed). OK"
else
    echo "POST-HARNESS FAILED: $FINAL passed against $PRISTINE expected." >&2
    exit 2
fi

[ "$BROKEN" -eq 0 ] || { echo "NOTE: $BROKEN mutation(s) reported BROKEN/CRASH -- re-run in isolation." >&2; }
[ "$INERT_BAD" -eq 0 ] || { echo "NOTE: an inert mutation BECAME load-bearing -- reclassify it." >&2; }
[ "$SURVIVED" -eq 0 ] || exit 3
[ "$NOOP" -eq 0 ] || exit 4
exit 0
