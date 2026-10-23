#!/usr/bin/env bash
# =============================================================================
# tests/mutate_logit_processing.sh -- mutation harness for
# include/nn/utils/logit_processing.{h,cpp}
#
# It is tests/mutate_sinkhorn.sh with the file paths and mutation list swapped
# out. That is deliberate, and NOT a violation of the TDD "delete means delete"
# rule: the HARNESS is not production code, it is this repo's accumulated
# knowledge about how a harness lies. Every defence below was written in
# response to a real false verdict in an earlier session, and re-deriving them
# would mean re-earning the same bug.
#
#   1. `restore()` MUST NEVER DELETE THE BACKUP. It is called once PER MUTATION
#      as well as on EXIT, so a self-cleaning restore works exactly once and
#      every later restore is a silent no-op -- the tree is left holding
#      whatever mutant ran last. (The hutchinson session: the post-harness
#      rebuild reported "23 passed / 17 failed" against a tree nobody edited.)
#   2. The link set comes from `make print-lib-objs`, NEVER from regex-parsing
#      the Makefile. `LIB_OBJS` is a RECURSIVE make variable whose literal text
#      contains no `.o` filenames, so a regex yields an empty set and every
#      mutation scores MUT-BROKEN -- one defect reported as N verdicts.
#   3. The MUTATED OBJECT IS EXCLUDED from the link set. A link error is not a
#      test failure.
#   4. BOTH the .cpp and the header are snapshotted and restored ATOMICALLY.
#      A plain `cp` truncates the destination first, so a concurrent reader can
#      splice two generations together (this actually happened once).
#   5. `rm -f` intermediates before every build (stale .o -> phantom failures).
#   6. EVERY EDIT IS VERIFIED BY CHECKSUM, and an ANCHOR THAT MATCHES MORE THAN
#      ONCE IS REFUSED rather than silently editing the first match. The
#      sinkhorn harness's first run reported MUT-NOOP for two mutations for
#      exactly this reason; a refusal is correct behaviour.
#   7. MUT-CRASH gets its OWN column. A missing sanitizer binary is UNAVAILABLE,
#      never a verdict.
#   8. The tree is rebuilt and re-verified ON EXIT before any verdict is
#      reported.
#   9. A SELF-CHECK runs a known-bad mutation FIRST. A harness that has never
#      detected anything is not evidence of a weak suite; it is evidence of a
#      broken harness.
#  10. MUT-BROKEN means "the build failed", NOT "the mutation is bad" -- and
#      neither implies the other. A suspicious MUT-BROKEN is re-run in
#      isolation before being believed.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC_CPP="include/nn/utils/logit_processing.cpp"
SRC_HDR="include/nn/utils/logit_processing.h"
BIN="build/test_logit_processing"
OBJ="build/utils/logit_processing.o"
TOBJ="build/test_logit_processing.o"

BAK_CPP="$(mktemp)"
BAK_HDR="$(mktemp)"
cp "$SRC_CPP" "$BAK_CPP"
cp "$SRC_HDR" "$BAK_HDR"

restore() {
    # ATOMIC: a plain `cp` truncates the destination first, so a concurrent
    # reader can splice two generations together.
    cp "$BAK_CPP" "$SRC_CPP.tmp" && mv -f "$SRC_CPP.tmp" "$SRC_CPP"
    cp "$BAK_HDR" "$SRC_HDR.tmp" && mv -f "$SRC_HDR.tmp" "$SRC_HDR"
}
SCRATCH="$(mktemp -d)"
cleanup() {
    restore
    rm -f "$BAK_CPP" "$BAK_HDR"
    rm -rf "$SCRATCH"
    # THE POST-HARNESS REBUILD-AND-VERIFY DEFENCE. An earlier session shipped a
    # restored source with a stale mutant still on disk and reported a phantom
    # regression; this is the check that catches it.
    rm -f "$OBJ" "$TOBJ" "$BIN"
    make "$BIN" >/dev/null 2>&1
    if "$BIN" >/dev/null 2>&1; then
        echo ""
        echo "POST-HARNESS: tree restored, rebuilt, and GREEN."
    else
        echo ""
        echo "POST-HARNESS FAILURE: the restored tree does not pass. STOP."
        "$BIN" 2>&1 | tail -20
        exit 1
    fi
}
trap cleanup EXIT INT TERM

checksum() { md5sum "$1" | cut -d' ' -f1; }

# ---------------------------------------------------------------------------
# Link set, from make itself. Assert the count so an empty extraction aborts
# BEFORE any verdict is reported rather than scoring every mutation MUT-BROKEN.
# ---------------------------------------------------------------------------
mapfile -t ALL_OBJS < <(make print-lib-objs)
if [ "${#ALL_OBJS[@]}" -lt 10 ]; then
    echo "ABORT: link set has only ${#ALL_OBJS[@]} objects (need >= 10)."
    exit 1
fi
LINK_OBJS=()
for o in "${ALL_OBJS[@]}"; do
    case "$o" in
        *logit_processing.o|*test_logit_processing.o) ;;   # exclude mutated units
        *) LINK_OBJS+=("$o") ;;
    esac
done
echo "link set: ${#LINK_OBJS[@]} objects (excluded the mutated units)"

# ---------------------------------------------------------------------------
run_suite() {
    stdbuf -o0 "$BIN" 2>/dev/null | grep -E '=== Summary: [0-9]+ passed, [0-9]+ failed ===' \
        | tail -1
}
parse_failed() { echo "$1" | sed -E 's/.* ([0-9]+) failed ===/\1/'; }

CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0; INERT_CONFIRMED=0
declare -a RESULTS
record() { RESULTS+=("$1|$2|$3"); }

apply_mutation() {  # $1 = file tag (cpp|hdr), $2 = replacement spec (old <US> new)
    python3 - "$1" "$2" <<'PYEOF'
import sys, pathlib
tag, spec = sys.argv[1], sys.argv[2]
path = {"cpp": "include/nn/utils/logit_processing.cpp",
        "hdr": "include/nn/utils/logit_processing.h"}[tag]
old, new = spec.split("\x1f")
p = pathlib.Path(path)
s = p.read_text()
if s.count(old) != 1:
    print(f"ANCHOR_COUNT={s.count(old)}")
    sys.exit(3)          # 0 = no-op, >1 = ambiguous -- BOTH are harness defects
p.write_text(s.replace(old, new))
PYEOF
}

# ---------------------------------------------------------------------------
# SELF-CHECK: prove the harness can produce a CAUGHT verdict at all.
# ---------------------------------------------------------------------------
echo "=== SELF-CHECK: injecting a known-bad mutation (nucleus >= -> >) ==="
if apply_mutation cpp "$(printf 'if (cum >= p_threshold) break;\x1fif (cum > p_threshold) break;')"; then
    rm -f "$OBJ" "$TOBJ" "$BIN"
    make "$BIN" >/dev/null 2>&1
    line="$(run_suite)"
    if [ -z "$line" ]; then
        echo "SELF-CHECK: MUT-CRASH (no summary line) -- harness counting is suspect"
    elif [ "$(parse_failed "$line")" -gt 0 ]; then
        echo "SELF-CHECK: OK -- the suite detects a broken nucleus cut ($line)"
    else
        echo "SELF-CHECK FAILED: the suite passed a deliberately broken nucleus."
        exit 1
    fi
else
    echo "SELF-CHECK FAILED: could not apply the self-check mutation."
    exit 1
fi
restore
rm -f "$OBJ" "$TOBJ" "$BIN"
make "$BIN" >/dev/null 2>&1
echo "self-check restored; pristine line: $(run_suite)"
echo ""

run_mutation() {  # $1=id $2=file tag $3=spec $4=description
    local id="$1" tag="$2" spec="$3" desc="$4"
    local before after rc
    before="$(checksum "$SRC_CPP")$(checksum "$SRC_HDR")"

    apply_mutation "$tag" "$spec"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        restore
        if [ "$rc" -eq 3 ]; then
            echo "  $id MUT-NOOP/AMBIGUOUS -- harness defect, not a verdict: $desc"
        else
            echo "  $id MUT-NOOP (edit did not apply) -- harness defect: $desc"
        fi
        NOOP=$((NOOP+1)); record "$id" "MUT-NOOP" "$desc"; return
    fi

    after="$(checksum "$SRC_CPP")$(checksum "$SRC_HDR")"
    if [ "$before" = "$after" ]; then
        restore
        echo "  $id MUT-NOOP (checksum unchanged) -- harness defect: $desc"
        NOOP=$((NOOP+1)); record "$id" "MUT-NOOP" "$desc"; return
    fi

    rm -f "$OBJ" "$TOBJ" "$BIN"
    if ! make "$BIN" >/dev/null 2>&1; then
        # A BUILD failure is not a verdict. Re-run in isolation to be sure this
        # is the mutation and not a concurrent build race.
        restore; rm -f "$OBJ" "$TOBJ" "$BIN"
        if make "$BIN" >/dev/null 2>&1; then
            echo "  $id MUT-BROKEN transient (build raced, pristine builds fine): $desc"
            BROKEN=$((BROKEN+1)); record "$id" "MUT-BROKEN" "$desc"; return
        fi
        echo "  $id MUT-BROKEN (build failed, persists in isolation): $desc"
        BROKEN=$((BROKEN+1)); record "$id" "MUT-BROKEN" "$desc"; return
    fi

    local line n
    line="$(run_suite)"

    # RESTORE AFTER EVERY MUTATION, and rebuild the pristine binary.
    #
    # The upstream harness only restores on the MUT-BROKEN and MUT-NOOP paths,
    # relying on the NEXT mutation's `replace` to overwrite the previous one.
    # That works by accident while every mutation touches the same file, and it
    # silently breaks the moment one of them does NOT: the tree keeps the
    # previous mutant, the following mutation's anchor may not match (reported
    # as a bogus MUT-NOOP), and the final verdict column is computed against a
    # tree nobody believes in. Cost here: M03 and M12 both reported MUT-NOOP
    # against anchors that provably exist exactly once.
    rm -f "$OBJ" "$TOBJ" "$BIN"
    make "$BIN" >/dev/null 2>&1
    restore

    if [ -z "$line" ]; then
        echo "  $id CAUGHT (crash, no summary) -- the detector worked, the counting did not: $desc"
        CRASH=$((CRASH+1)); record "$id" "CAUGHT-CRASH" "$desc"; return
    fi
    n="$(parse_failed "$line")"
    if [ "$n" -gt 0 ]; then
        echo "  $id CAUGHT ($n assertions): $desc"
        CAUGHT=$((CAUGHT+1)); record "$id" "CAUGHT($n)" "$desc"; return
    fi
    echo "  $id *** SURVIVED *** $line : $desc"
    SURVIVED=$((SURVIVED+1)); record "$id" "SURVIVED" "$desc"
}

# ---------------------------------------------------------------------------
# run_mutation_expect_inert -- the INVERTED verdict.
#
# A survivor that is not explained is indistinguishable from a real coverage
# gap, so an inert mutation gets its own function that PASSES when the suite
# stays green under the mutation and FAILS when the mutation starts to matter.
# That way a mutation which was inert yesterday and becomes load-bearing after
# a future refactor is caught automatically instead of being absorbed into a
# silently-growing pile of "survivors".
# ---------------------------------------------------------------------------
run_mutation_expect_inert() {  # $1=id $2=file tag $3=spec $4=description
    local id="$1" tag="$2" spec="$3" desc="$4"
    local before after rc
    before="$(checksum "$SRC_CPP")$(checksum "$SRC_HDR")"

    apply_mutation "$tag" "$spec"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        restore
        echo "  $id INERT-CHECK HARNESS DEFECT (edit did not apply): $desc"
        NOOP=$((NOOP+1)); record "$id" "HARNESS-DEFECT" "$desc"; return
    fi
    after="$(checksum "$SRC_CPP")$(checksum "$SRC_HDR")"
    if [ "$before" = "$after" ]; then
        restore
        echo "  $id INERT-CHECK HARNESS DEFECT (checksum unchanged): $desc"
        NOOP=$((NOOP+1)); record "$id" "HARNESS-DEFECT" "$desc"; return
    fi

    rm -f "$OBJ" "$TOBJ" "$BIN"
    if ! make "$BIN" >/dev/null 2>&1; then
        restore; rm -f "$OBJ" "$TOBJ" "$BIN"
        echo "  $id INERT-CHECK: the mutation does not even BUILD -- it is NOT inert"
        BROKEN=$((BROKEN+1)); record "$id" "NOT-INERT(build)" "$desc"; return
    fi

    local line
    line="$(run_suite)"
    # Rebuild the PRISTINE binary BEFORE restoring the source, exactly as
    # run_mutation does -- restoring first would leave the mutated object on
    # disk under a pristine header, which is the stale-.o phantom-failure mode.
    rm -f "$OBJ" "$TOBJ" "$BIN"
    make "$BIN" >/dev/null 2>&1
    restore

    if [ -z "$line" ]; then
        echo "  $id INERT-CHECK *** the mutation CRASHES the suite -- it is NOT inert ***"
        BROKEN=$((BROKEN+1)); record "$id" "NOT-INERT(crash)" "$desc"; return
    fi
    if [ "$(parse_failed "$line")" -gt 0 ]; then
        echo "  $id INERT-CHECK *** the mutation BREAKS the suite -- it is NOT inert ***"
        BROKEN=$((BROKEN+1)); record "$id" "NOT-INERT(caught)" "$desc"; return
    fi
    INERT_CONFIRMED=$((INERT_CONFIRMED+1))
    record "$id" "INERT-CONFIRMED" "$desc"
    echo "  $id INERT-CONFIRMED (suite stays green, as measured): $desc"
}

echo "=== MUTATIONS ==="
run_mutation M01 cpp \
 "$(printf 'if (cum >= p_threshold) break;\x1fif (cum > p_threshold) break;')" \
 "nucleus uses a STRICT inequality -- the off-by-one; admits one token too many"

# --- S5: the repetition-penalty sign split. THE other headline mutation. ----
# Two of these, one per branch, because deleting one branch is a different bug
# from inverting the comparison and both are plausible slips.
run_mutation M02 cpp \
 "$(printf 'set_at(out, i, x < 0.0 ? x * penalty : x / penalty);\x1fset_at(out, i, x * penalty);')" \
 "repetition penalty always multiplies (INVERTS the penalty on positive logits)"

run_mutation M03 cpp \
 "$(printf 'set_at(out, i, x < 0.0 ? x %s penalty : x / penalty);\x1fset_at(out, i, x >= 0.0 ? x / penalty : x %s 1.0);' '*' '*')" \
 "repetition penalty negative branch neutralised (only positives suppressed)"

# --- S6: frequency vs presence must differ ---------------------------------
run_mutation M04 cpp \
 "$(printf 'set_at(out, j, x - penalty * count[j]);\x1fset_at(out, j, x - penalty);')" \
 "frequency penalty de-linearised (becomes identical to presence)"

# --- S4: min-p threshold is relative to p_MAX ------------------------------
run_mutation M05 cpp \
 "$(printf 'double thr = min_p * max_prob(probs);\x1fdouble thr = min_p * probs[probs.size() - 1];')" \
 "min-p threshold relative to the LAST probability instead of the max"

# --- S7: temperature acts on logits -----------------------------------------
run_mutation M06 cpp \
 "$(printf 'set_at(out, j, std::isinf(x) ? x : x / temperature);\x1fset_at(out, j, std::isinf(x) ? x : std::log(std::exp(x) / temperature));')" \
 "temperature applied to the softmax OUTPUT instead of the logits"

# --- S10: the max-subtract before exp --------------------------------------
run_mutation M07 cpp \
 "$(printf 'double m = *std::max_element(logits.begin(), logits.end());\x1fdouble m = 0.0;')" \
 "softmax max-subtract neutralised (exp overflows to inf, inf/inf is NaN)"

# --- S11: std::round vs nearbyint ------------------------------------------
run_mutation M08 cpp \
 "$(printf 'set_at(out, j, std::round(x));\x1fset_at(out, j, std::nearbyint(x));')" \
 "round_to_int uses nearbyint (ties-to-even instead of away-from-zero)"

# --- S12: first-wins tie-break in greedy -----------------------------------
run_mutation M09 cpp \
 "$(printf 'if (lg[j] > lg[best]) best = j;\x1fif (lg[j] >= lg[best]) best = j;')" \
 "greedy argmax tie-break to LAST wins instead of first"

# --- S12: first-wins tie-break in the sort ---------------------------------
run_mutation M10 cpp \
 "$(printf 'return v[a] > v[b]; });\x1freturn v[a] >= v[b]; });')" \
 "descending_order comparator is not strict (an unstable, UB-violating sort)"

# --- top-k k >= V is a no-op -----------------------------------------------
# MEASURED INERT, and the reason is structural rather than accidental: at
# k == V the mutant's `k > v` branch is skipped, but the write-back loop
# immediately below still runs for every index, so every entry is restored to
# its original value. The observable result is bit-identical. Scored with the
# INVERTED verdict, which fails if a future refactor ever makes it matter.
run_mutation_expect_inert M11 cpp \
 "$(printf 'if (k >= v) {\x1fif (k > v) {')" \
 "top-k k >= V no-op turned into k > V -- MEASURED INERT: at k == V the "\
"write-back loop still restores every index, so the output is identical"

# --- S14: the length penalty belongs ONLY in the final ranking --------------
# Inverting this one is the interesting case: moving the penalty into the live
# expansion does not change the RESULT on a uniform fixture, so this is the
# mutation most likely to survive and most worth running.
run_mutation M14 cpp \
 "$(printf 'sa /= std::pow(la, cfg_.length_penalty);\x1fsa /= std::pow(la, cfg_.length_penalty) + 1.0;')" \
 "length penalty offset by 1.0 (a different normaliser)"

# --- beam search: eod is not eos -------------------------------------------
# This one SURVIVED the first harness run and the reason was a missing test,
# not an inert mutation: the suite checked that an eod sentinel appeared in the
# retained hypotheses, but never that DECODING CONTINUED PAST one. Making the
# last vocabulary id also terminate was therefore invisible.
#
# The discriminator: a model that emits the last id on every step. Correct
# behaviour keeps generating (the hypotheses stay short and the returned tensor
# spans max_length tokens); the mutant stops at the first one. Test 15 pins both
# directions -- 15b that a non-eos sentinel does not stop it, and 15d that a
# real eos does.
run_mutation M15 cpp \
 "$(printf 'if (c.token == cfg_.eos_id) {\x1fif (c.token == cfg_.eos_id || c.token == static_cast<int>(vocab_) - 1) {')" \
 "the LAST vocabulary id is ALSO treated as eos (conflates eod with eos)"

# --- the -inf guard in apply_temperature is algebraically inert -------------
# `-inf / T == -inf` for every finite T > 0 in IEEE-754, so the explicit
# `std::isinf(x) ? x : ...` guard changes no observable value. It is kept in the
# source as documentation, not as computation. Also scored with the inverted
# verdict.
run_mutation_expect_inert M12 cpp \
 "$(printf 'set_at(out, j, std::isinf(x) ? x : x / temperature);\x1fset_at(out, j, x / temperature);')" \
 "apply_temperature drops its explicit -inf guard -- MEASURED INERT: "\
"-inf/T == -inf in IEEE-754 for every finite T > 0"

# --- beam search: batch contract is enforced -------------------------------
# The replacement stays BRACE-BALANCED. An earlier draft wrote `if (false)` in
# place of the condition, which deleted the closing brace and reported
# MUT-BROKEN -- a BUILD verdict for a mutation that was perfectly well formed.
run_mutation M16 cpp \
 "$(printf 'if (next.rows != live.size() || next.cols != vocab_) {\x1fif (next.rows != live.size() + 1000000 || next.cols != vocab_) {')" \
 "score_fn shape guard neutered (braces kept balanced)"

# --- history validation: out-of-range ids must throw -----------------------
run_mutation M17 cpp \
 "$(printf 'if (t < 0 || static_cast<size_t>(t) >= vocab) {\x1fif (t < 0 || static_cast<size_t>(t) >= 1000000) {')" \
 "history token-id range check neutered"

# --- the sampler must never leave a truncated support ------------------------
run_mutation M18 cpp \
 "$(printf 'if (u < acc) return static_cast<int>(j);\x1fif (u < acc || j == 0) return static_cast<int>(j);')" \
 "multinomial CDF falls back to index 0 on a truncated support"

# --- S8: the chain folds LEFT TO RIGHT -------------------------------------
run_mutation M13 cpp \
 "$(printf 'for (const auto& fn : processors_) cur = fn(cur);\x1ffor (auto it = processors_.rbegin(); it != processors_.rend(); ++it) cur = (*it)(cur);')" \
 "chain applied RIGHT TO LEFT (the documented order is left to right)"

# --- softmax normalises ----------------------------------------------------
run_mutation M19 cpp \
 "$(printf 'p[i] /= sum;\x1fp[i] *= sum;')" \
 "softmax divides by the wrong quantity"

# --- top-p accumulates the PROBABILITY, not the logit ----------------------
# The anchor is now LONGER because the one-line form became ambiguous when
# typical-p was added: `cum += probs[idx];` appears in BOTH apply_top_p and
# apply_typical_p, so the harness would have refused the edit (and correctly
# so -- it edits ONE occurrence, and which one it picked would be a coin
# flip). The form anchored on `descending_order(lg)` is unique to apply_top_p,
# because typical-p sorts by DISTANCE with its own comparator.
run_mutation M20 cpp \
 "$(printf '    std::vector<size_t> order = descending_order(lg);\n\n    double cum = 0.0;\n    for (size_t i = 0; i < order.size(); ++i) {\n        size_t idx = order[i];\n        set_at(out, idx, get_at(logits, idx));\n        cum += probs[idx];\x1f    std::vector<size_t> order = descending_order(lg);\n\n    double cum = 0.0;\n    for (size_t i = 0; i < order.size(); ++i) {\n        size_t idx = order[i];\n        set_at(out, idx, get_at(logits, idx));\n        cum += lg[idx];')" \
 "nucleus accumulates LOGITS instead of probabilities"

# --- min-p must apply the temperature BEFORE the threshold -----------------
# This one survived the first two harness runs. The suite DID assert something
# about the temperature (`n_finite(T=2) >= n_finite(T=1)`), but an inequality
# holds whether or not the temperature is honoured at all -- it was testing
# monotonicity of a number, not the behaviour. The fixture below asserts exact
# survivor SETS at T = 0.5, 1.0 and 4.0, which is what killed it.
run_mutation M21 cpp \
 "$(printf 'std::vector<double> lg = to_vec(logits);\n    for (size_t j = 0; j < lg.size(); ++j) {\n        if (!std::isinf(lg[j])) lg[j] /= temperature;\n    }\n    std::vector<double> probs = probabilities(lg);\n\n    // The floor is min_p * p_max\x1fstd::vector<double> lg = to_vec(logits);\n    std::vector<double> probs = probabilities(lg);\n\n    // The floor is min_p * p_max')" \
 "min-p ignores its temperature argument"

# ---------------------------------------------------------------------------
# HARNESS DEFECT FOUND BY ADDING A FEATURE (the twenty-second instance of
# this class, and the first one caused by a NEW feature rather than by a new
# mutation).
#
# The anchor `if (!std::isinf(lg[j])) lg[j] /= temperature;` matched ONCE
# before typical-p was added. `apply_typical_p` contains THE SAME LINE, so the
# count is now TWO and the harness REFUSES to edit an ambiguous anchor --
# correctly: it would have edited whichever it found first, which is not a
# verdict but a coin flip.
#
# The fix is a three-part anchor that is unique by construction: the closing
# brace of the loop, the `probabilities(lg)` call, and the comment that
# follows it. The replacement keeps the first two parts and drops the third,
# so the MUTATED min-p skips its temperature division while typical-p keeps
# its own. The MUTATION IS STILL THE MUTATION.
#
# THE LESSON, which generalises: adding a feature can INVALIDATE an existing
# mutation's anchor, and the failure is silent in the sense that the harness
# reports a harness defect rather than a verdict -- so a reader who skims the
# output would see "M21 MUT-NOOP/AMBIGUOUS" and might read it as "M21 is
# inert". It is not inert; the edit simply was not applied. Check the anchor
# count after EVERY feature addition, not only after every mutation.
# ---------------------------------------------------------------------------

# ============================================================================
# TYPICAL-P (S15..S21) -- the mutations added with the feature.
#
# M23 IS THE HEADLINE. The sort direction is the single most consequential
# decision in the whole algorithm, and a SURVIVOR-COUNT test cannot see it: the
# descending variant still returns a non-empty set of a plausible size, so the
# count looks right and only the SET is wrong. The suite pins exact survivor
# sets, which is what kills it. This is the same lesson the repo has already
# learned twice -- the S8 chain check "compared two hand-composed calls" and
# the min-p check "asserted an inequality, which holds whether or not the
# temperature is honoured at all".
# ============================================================================
run_mutation M22 cpp \
 "$(printf '? std::fabs(-std::log(probs[j]) - entropy)\x1f? std::fabs(std::log(probs[j]) - entropy)')" \
 "typical-p: the log-probability is NOT negated in the distance (S15)"

run_mutation M23 cpp \
 "$(printf 'return dist[a] < dist[b]; });\x1freturn dist[a] > dist[b]; });')" \
 "typical-p: the sort is DESCENDING by distance -- the LEAST typical first (S16)"

run_mutation M24 cpp \
 "$(printf 'if (cum >= tau) break;\x1fif (cum > tau) break;')" \
 "typical-p: the cut is a STRICT inequality -- the crossing token is DROPPED (S17)"

run_mutation M25 cpp \
 "$(printf 'if (probs[j] > 0.0) entropy -= probs[j] * std::log(probs[j]);\x1fif (probs[j] > 0.0) entropy += probs[j] * std::log(probs[j]);')" \
 "typical-p: the entropy's SIGN is flipped (S18)"

run_mutation M26 cpp \
 "$(printf 'if (probs[j] > 0.0) entropy -= probs[j] * std::log(probs[j]);\x1fentropy -= probs[j] * std::log(probs[j]);')" \
 "typical-p: the zero-probability guard is dropped from the entropy sum (S18)"

run_mutation M27 cpp \
 "$(printf ': std::numeric_limits<double>::infinity();\x1f: 0.0;')" \
 "typical-p: a zero-probability token gets distance 0 instead of infinity (S20)"

# The typical-p TEMPERATURE is a SEPARATE code path from min-p's, so it gets
# its own mutation rather than reusing M21. The anchor is the two comment
# lines above the loop, which are unique to apply_typical_p.
run_mutation M29 cpp \
 "$(printf '    // The temperature acts on the LOGITS FIRST, so the entropy below is that\n    // of the temperature-adjusted distribution.\n    for (size_t j = 0; j < lg.size(); ++j) {\n        if (!std::isinf(lg[j])) lg[j] /= temperature;\n    }\x1f    // The temperature acts on the LOGITS FIRST, so the entropy below is that\n    // of the temperature-adjusted distribution.')" \
 "typical-p: ignores its temperature argument (the entropy is then measured on \
the UNADJUSTED distribution)"

# The temperature VALIDATION guard. A plausible slip: dropping the finiteness
# check and letting a NaN temperature through.
run_mutation M30 cpp \
 "$(printf 'if (!(temperature > 0.0) || !std::isfinite(temperature)) {\n        throw std::invalid_argument(\n            \"apply_typical_p: temperature must be finite and > 0\");\n    }\x1fif (!(temperature > 0.0)) {\n        throw std::invalid_argument(\n            \"apply_typical_p: temperature must be finite and > 0\");\n    }')" \
 "typical-p: the temperature's FINITENESS check is dropped (a NaN temperature \
is accepted)"

# The tau RANGE guard.
run_mutation M31 cpp \
 "$(printf 'if (tau < 0.0 || tau > 1.0) {\n        throw std::invalid_argument(\"apply_typical_p: tau must be in [0, 1]\");\n    }\x1fif (tau < 0.0 || tau > 1.5) {\n        throw std::invalid_argument(\"apply_typical_p: tau must be in [0, 1]\");\n    }')" \
 "typical-p: the tau range guard is loosened (tau = 1.25 is wrongly accepted)"

# ---------------------------------------------------------------------------
echo ""
echo "=== ASAN+UBSAN LANE (inverted verdict: it FAILS if the PRISTINE build is not clean) ==="
ASAN_OK="UNAVAILABLE"
mkdir -p "$SCRATCH/asan"
if g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude \
       -c "$SRC_CPP" -o "$SCRATCH/asan/lp.o" 2>/dev/null &&
   g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude \
       -c tests/test_logit_processing.cpp -o "$SCRATCH/asan/test.o" 2>/dev/null; then
    # Consistent flags on BOTH TUs: mixing -O1 instrumented and -O2 library
    # objects makes inlined constants disagree and produces phantom failures.
    if g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude \
           "$SCRATCH/asan/lp.o" "$SCRATCH/asan/test.o" \
           "${LINK_OBJS[@]}" -o "$SCRATCH/asan/test_lp" 2>/dev/null; then
        if ASAN_OPTIONS=detect_leaks=0 "$SCRATCH/asan/test_lp" \
             >"$SCRATCH/asan/out.txt" 2>&1; then
            ASAN_OK="CLEAN"
            echo "  ASan+UBSan on the PRISTINE tree: CLEAN"
        else
            echo "  ASan+UBSAN FAILURE on the pristine tree:"
            tail -20 "$SCRATCH/asan/out.txt"
        fi
    else
        echo "  ASan link failed -- reporting UNAVAILABLE, never a verdict"
    fi
else
    echo "  ASan compile failed -- reporting UNAVAILABLE, never a verdict"
fi

echo ""
echo "============================================================"
echo " MUTATION SUMMARY: $CAUGHT caught / $SURVIVED survived / $BROKEN broken / $CRASH crash / $NOOP noop / $INERT_CONFIRMED inert-confirmed"
echo " ASAN pristine build: $ASAN_OK"
echo "============================================================"
for r in "${RESULTS[@]}"; do echo "   $r"; done