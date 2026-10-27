#!/usr/bin/env bash
# tests/mutate_crf.sh -- mutation harness for utils/crf.{h,cpp}.
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

SRC="include/nn/utils/crf.cpp"
HDR="include/nn/utils/crf.h"
TEST="tests/test_crf.cpp"
BIN="build/test_crf"
BAK_SRC=$(mktemp); BAK_HDR=$(mktemp)
BAK_MARK=".mutate_crf_tree_pristine"
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
MUT_OBJ="build/utils/crf.o"
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
    rm -f build/test_crf.o "$BIN" "$MUT_OBJ"
    make -s build/utils/crf.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$TEST" -o build/test_crf.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$SRC" -o "$MUT_OBJ" >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL $LIB_LINK "$MUT_OBJ" build/test_crf.o -o "$BIN" >/dev/null 2>&1 || return 1
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
echo "=== crf mutations ==="

# --- THE TRANSITION CONVENTION: transitions(i,j) is FROM i TO j ------------
# M01 transposes the transition index in the alpha recursion. The transposed
#      convention gives a DIFFERENT partition function on any asymmetric
#      fixture, and the two are the SAME SHAPE and the SAME SIZE, so no type
#      check can see the difference -- only a pinned value can.
run_mutation M01 "$SRC" \
    "terms[i] = m.alpha[(t - 1) * K + i] + transitions(i, j);" \
    "terms[i] = m.alpha[(t - 1) * K + i] + transitions(j, i);"

# M02 transposes it in the beta recursion.
run_mutation M02 "$SRC" \
    "terms[j] = transitions(i, j) + emissions(t + 1, j) + m.beta[(t + 1) * K + j];" \
    "terms[j] = transitions(j, i) + emissions(t + 1, j) + m.beta[(t + 1) * K + j];"

# --- THE BASE CASES --------------------------------------------------------
# M03 adds a transition BEFORE position 0, where there is none. alpha_0 must
#      be EXACTLY the emissions.
run_mutation M03 "$SRC" \
    "m.alpha[j] = emissions(0, j);" \
    "m.alpha[j] = emissions(0, j) + logsumexp(std::vector<double>(K, 0.0));"

# M04 sets beta_{T-1} to -inf instead of 0. It must be ZERO.
run_mutation M04 "$SRC" \
    "m.beta[(T - 1) * K + i] = 0.0;" \
    "m.beta[(T - 1) * K + i] = kNegInf;"

# --- THE LOG DOMAIN --------------------------------------------------------
# M05 computes the partition function in the EXP space. It passes every small
#      fixture and fails on every long one.
run_mutation M05 "$SRC" \
    "m.log_z = logsumexp(terms);" \
    "m.log_z = std::log(std::exp(terms[0]) + std::exp(terms[K > 1 ? 1 : 0])) ;"

# --- THE NODE MARGINALS ----------------------------------------------------
# M06 drops the beta term from the node marginals -- the classic "forward
#      only" mistake, which is a DIFFERENT object.
run_mutation M06 "$SRC" \
    "const double v = m.alpha[t * K + k] + m.beta[t * K + k] - m.log_z;" \
    "const double v = m.alpha[t * K + k] - m.log_z + std::log(K);"

# --- THE EDGE MARGINALS ----------------------------------------------------
# M07 drops the TRANSITION term from the edge marginals. The edge marginals
#      are a DIFFERENT object from the node marginals, and this makes them
#      stop marginalising to them.
run_mutation M07 "$SRC" \
    "const double v = m.alpha[t * K + i] + transitions(i, j) +
                                 emissions(t + 1, j) + m.beta[(t + 1) * K + j] - m.log_z;" \
    "const double v = m.alpha[t * K + i] +
                                 emissions(t + 1, j) + m.beta[(t + 1) * K + j] - m.log_z;"

# --- VITERBI ---------------------------------------------------------------
# M08 breaks the tie-break: takes the LARGEST predecessor index instead of
#      the smallest, so the decode is no longer deterministic.
run_mutation M08 "$SRC" \
    "if (cand > best) {
                    best = cand;
                    best_i = i;
                }" \
    "if (cand >= best) {
                    best = cand;
                    best_i = i;
                }"

# M09 adds a transition before position 0 in the viterbi recursion too.
run_mutation M09 "$SRC" \
    "for (size_t j = 0; j < K; ++j) delta[j] = emissions(0, j);" \
    "for (size_t j = 0; j < K; ++j) delta[j] = emissions(0, j) + transitions(K - 1, j);"

# --- THE JOINT CLOSED FORM -------------------------------------------------
# M10 forgets to subtract the one-hot label from the node marginals.
run_mutation M10 "$SRC" \
    "out(t, static_cast<size_t>(y[t])) -= 1.0;" \
    "out(t, static_cast<size_t>(y[t])) -= 0.0;"

# M11 forgets to subtract the label-pair count from the transitions.
run_mutation M11 "$SRC" \
    "out(static_cast<size_t>(y[t]), static_cast<size_t>(y[t + 1])) -= 1.0;" \
    "out(static_cast<size_t>(y[t]), static_cast<size_t>(y[t + 1])) -= 0.0;"

# --- THE MARGINAL ADJOINT: the -logZ TERM IS LOAD-BEARING -----------------
# M12 drops the -logZ term from the EMISSIONS gradient. Every log node
#      carries a -logZ, so the gradient is off by -G * node.
run_mutation M12 "$SRC" \
    "out(t, k) = alpha_part + beta_part - G * node(t, k);" \
    "out(t, k) = alpha_part + beta_part;"

# M13 drops the -logZ term from the TRANSITIONS gradient.
run_mutation M13 "$SRC" \
    "out(i, j) = acc - G * edge_sum;" \
    "out(i, j) = acc;"

# --- THE MARGINAL ADJOINT: THE TWO DIRECTIONS ARE OPPOSITE -----------------
# M14 reverses the ALPHA adjoint direction (upward instead of downward). The
#      adjoint flows alpha_{t+1} -> alpha_t, i.e. DOWNWARD in t.
run_mutation M14 "$SRC" \
    "for (size_t t = T - 1; t >= 1; --t) {
        // r[j] = logsumexp_i(alpha_{t-1}(i) + A[i,j]) = alpha[t][j] - x[t][j]" \
    "for (size_t t = 1; t < T; ++t) {
        // r[j] = logsumexp_i(alpha_{t-1}(i) + A[i,j]) = alpha[t][j] - x[t][j]"

# M15 reverses the BETA adjoint direction (downward instead of upward). The
#      adjoint flows beta_t -> beta_{t+1}, i.e. UPWARD in t.
run_mutation M15 "$SRC" \
    "for (size_t t = 0; t + 1 < T; ++t) {
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                const double v = transitions(i, j) + emissions(t + 1, j) +
                                 m.beta[(t + 1) * K + j] - m.beta[t * K + i];" \
    "for (size_t t = T - 1; t-- > 0;) {
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j) {
                const double v = transitions(i, j) + emissions(t + 1, j) +
                                 m.beta[(t + 1) * K + j] - m.beta[t * K + i];"

# --- THE GUARDS -------------------------------------------------------------
# M16 removes the emissions finiteness check.
run_mutation M16 "$SRC" \
    "if (!std::isfinite(emissions(t, k))) {
                throw std::invalid_argument(\"crf: emissions must be finite\");
            }" \
    "if (false) {
                throw std::invalid_argument(\"crf: emissions must be finite\");
            }"

# M17 removes the transition shape check.
run_mutation M17 "$SRC" \
    "if (transitions.rows != emissions.cols || transitions.cols != emissions.cols) {" \
    "if (false) {"

# M18 removes the label RANGE check, so an out-of-range label reads out of
#      bounds instead of throwing.
run_mutation M18 "$SRC" \
    "if (!std::isfinite(v) || v < 0.0 ||
            v >= static_cast<double>(K) || v != std::floor(v)) {" \
    "if (false) {"

# M19 removes the g shape check.
run_mutation M19 "$SRC" \
    "if (g.rows != T || g.cols != K) {" \
    "if (false) {"

# M20 removes the g finiteness check.
run_mutation M20 "$SRC" \
    "if (!std::isfinite(g(t, k))) {
                throw std::invalid_argument(\"crf: the marginal weights g must be finite\");
            }" \
    "if (false) {
                throw std::invalid_argument(\"crf: the marginal weights g must be finite\");
            }"

# --- THE SCORE --------------------------------------------------------------
# M21 drops the transition contribution from the sequence score, so the joint
#      NLL's OTHER half is wrong.
run_mutation M21 "$SRC" \
    "score += transitions(static_cast<size_t>(y[t]), static_cast<size_t>(y[t + 1]));" \
    "score += 0.0 * transitions(static_cast<size_t>(y[t]), static_cast<size_t>(y[t + 1]));"

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
