#!/usr/bin/env bash
# Mutation harness for tests/test_tta.cpp.
#
# Carries every defence earlier sessions in this repo had to learn the hard way
# (see EXPANSION_QUEUE.md ## Done):
#   1. Link set comes from `make print-lib-objs`, never from parsing the
#      Makefile with a regex. LIB_OBJS is a RECURSIVE make variable whose
#      literal text contains a substitution reference and ZERO `.o` filenames,
#      so a regex either matches nothing or matches nothing useful -- and an
#      empty link set scores every real mutation "MUT-BROKEN", which would
#      "certify" a suite that catches nothing.
#   2. The object count is ASSERTED >= 10 and the run ABORTS before reporting
#      any verdict. One harness defect must never be reported as N verdicts.
#   3. The mutated TU is EXCLUDED from the link set (it is compiled separately).
#   4. Baseline is snapshotted once, before any mutation, and verified PRISTINE
#      by grepping markers that exist only in the correct implementation.
#   5. restore() is ATOMIC (cp to temp + mv -f), because a plain cp truncates
#      the destination and a concurrent reader can splice two generations.
#   6. `diff -q` after every sed, reporting MUT-NOOP -- an edit that never
#      applied is indistinguishable from a mutation that survived.
#   7. Compile failure is MUT-BROKEN, never MUT-CAUGHT. A link error is not a
#      test failure.
#   8. Crashes get their OWN verdict column (MUT-CRASH), counted separately.
#   9. Verdict is counted from [FAIL] lines, immune to summary-format changes.
#  10. Intermediates are rm -f'd before every build (stale .o false verdicts).
#  11. Grep uses -F where the anchor contains regex metacharacters.

set -uo pipefail
cd "$(dirname "$0")/.."

SRC=include/nn/utils/tta.cpp
HDR=include/nn/utils/tta.h
TESTBIN=build/test_tta
WORK=build/tta_mut
BASE=$(mktemp)

cleanup() {
    [ -f "$BASE" ] && cp -f "$BASE" "$SRC"
    rm -rf "$WORK"
    rm -f "$BASE" "${BASE}.h"
}
trap cleanup EXIT

# --- 4. baseline snapshot, asserted pristine -------------------------------
[ -f "$SRC" ] || { echo "HARNESS-DEFECT: $SRC missing"; exit 2; }
cp -f "$SRC" "$BASE"
cp -f "$HDR" "${BASE}.h"

pristine=1
for m in 'out(n, c) += p.w[t] * bundle(t, n * p.num_classes + c);' \
         'if (votes[c] > votes[best]) best = c;' \
         'out(n, c) = std::exp(acc(n, c) - mx) / sum;' \
         'p.w[i] = weights[i] / total;' \
         'out(r, i * width + (width - 1 - j)) = x(r, i * width + j);' \
         'if (t(i, j) > t(i, best)) best = j;' \
         'require(bundle.rows > 1,' \
         'softmax_row_into(bundle, t, n, p.num_classes, probs);'; do
    grep -qF "$m" "$SRC" || { echo "HARNESS-DEFECT: pristine marker absent: $m"; pristine=0; }
done
[ "$pristine" -eq 1 ] || { echo "ABORT: snapshot is not pristine"; exit 2; }

restore() {
    cp -f "$BASE" "$SRC.tmp.$$" && mv -f "$SRC.tmp.$$" "$SRC"
}

# --- 1 + 2. link set from make, with a count assertion ----------------------
mapfile -t LIB_OBJS < <(make print-lib-objs 2>/dev/null | grep '\.o$')
if [ "${#LIB_OBJS[@]}" -lt 10 ]; then
    echo "HARNESS-DEFECT: link set has ${#LIB_OBJS[@]} objects (need >= 10)."
    echo "ABORT before any verdict -- an empty link set would score every"
    echo "mutation MUT-BROKEN and certify a suite that catches nothing."
    exit 2
fi
# 3. drop the mutated TU from the link set
LINK_OBJS=()
for o in "${LIB_OBJS[@]}"; do
    [ "$o" = "build/utils/tta.o" ] && continue
    LINK_OBJS+=("$o")
done
echo "link set: ${#LINK_OBJS[@]} objects (tta.o excluded, compiled separately)"
echo

build_and_run() {
    rm -f "$WORK"/*.o "$TESTBIN"                      # 10. no stale objects
    mkdir -p "$WORK"
    g++ -std=c++17 -O2 -Wall -Wextra -march=native -Iinclude \
        -c "$SRC" -o "$WORK/tta.o" 2>"$WORK/cc.log" || { echo "MUT-BROKEN"; return; }
    grep -q 'warning:' "$WORK/cc.log" && { echo "MUT-BROKEN (new warning)"; return; }
    g++ -std=c++17 -O2 -Iinclude -c tests/test_tta.cpp -o "$WORK/test.o" 2>>"$WORK/cc.log" \
        || { echo "MUT-BROKEN"; return; }
    g++ "$WORK/tta.o" "$WORK/test.o" "${LINK_OBJS[@]}" -o "$TESTBIN" 2>>"$WORK/cc.log" \
        || { echo "MUT-BROKEN (link)"; return; }
    out=$("$TESTBIN" 2>&1); status=$?
    if [ $status -ge 128 ]; then echo "MUT-CRASH"; return; fi   # 8.
    fails=$(printf '%s\n' "$out" | grep -c '\[FAIL\]')            # 9.
    if [ "$fails" -gt 0 ]; then echo "MUT-CAUGHT ($fails assertions)"; else echo "MUT-SURVIVED"; fi
}

printf '%-4s %-58s %s\n' "ID" "MUTATION" "VERDICT"
printf '%s\n' "----------------------------------------------------------------------"

# Each spec is id|file|anchor|replacement, applied by `mut` below. `file` is
# SRC or HDR; anything else is refused by the spec guard rather than passed to
# an external program.
mut() {
    local id="$1" file="$2" anchor="$3" repl="$4"
    local target="$SRC" snapshot="$BASE"
    # Guard: refuse a spec whose file tag is not one we own. The adversarial
    # session's harness passed an `inert` marker through to `sed -i inert f.cpp`
    # and GNU sed read it as a SCRIPT, writing its diagnostics into the source.
    case "$file" in
        SRC) target="$SRC" ;;
        HDR) target="$HDR" ;;
        *) printf '%-4s %-58s %s\n' "$id" "bad file tag" "HARNESS-DEFECT"; return ;;
    esac
    # A spec with no anchor is a harness defect, never a verdict.
    if [ -z "$anchor" ] || [ -z "$repl" ]; then
        printf '%-4s %-58s %s\n' "$id" "empty spec" "HARNESS-DEFECT"; return
    fi
    if ! grep -qF "$anchor" "$target"; then
        printf '%-4s %-58s %s\n' "$id" "anchor not found: ${anchor:0:40}" "MUT-NOOP"; return
    fi
    local tmp="$target.mut.$$"
    cp -f "$target" "$tmp"
    python3 - "$tmp" "$anchor" "$repl" <<'PY'
import sys
p, a, r = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if a not in s:
    sys.exit(3)
open(p, 'w').write(s.replace(a, r, 1))
PY
    if [ $? -ne 0 ] || diff -q "$tmp" "$target" >/dev/null 2>&1; then
        printf '%-4s %-58s %s\n' "$id" "edit did not apply" "MUT-NOOP"; rm -f "$tmp"; return
    fi
    cp -f "$tmp" "$target"; rm -f "$tmp"
    local verdict; verdict=$(build_and_run)
    printf '%-4s %-58s %s\n' "$id" "$(echo "$anchor" | cut -c1-58)" "$verdict"
    if [ "$file" = "HDR" ]; then
        cp -f "${BASE}.h" "$HDR.tmp.$$" && mv -f "$HDR.tmp.$$" "$HDR"
    else
        restore
    fi
}

# M1  mean_logits: drop the per-transform weight -> becomes a plain SUM
mut M1 SRC 'out(n, c) += p.w[t] * bundle(t, n * p.num_classes + c);' \
           'out(n, c) += bundle(t, n * p.num_classes + c);'

# M2  mean_probs: divide by T AFTER averaging the weights (an off-by-T)
mut M2 SRC 'p.w.assign(p.num_transforms,
                   1.0 / static_cast<double>(p.num_transforms));' \
           'p.w.assign(p.num_transforms, 1.0);'

# M3  geo: DROP THE NORMALISATION (D3). Expected to be caught by VALUE
#     assertions and NOT by argmax assertions -- the whole point of D3.
mut M3 SRC 'out(n, c) = std::exp(acc(n, c) - mx) / sum;' \
           'out(n, c) = std::exp(acc(n, c) - mx);'

# M4  argmax tie-break: >= instead of > (D4)
mut M4 SRC 'if (t(i, j) > t(i, best)) best = j;' \
           'if (t(i, j) >= t(i, best)) best = j;'

# M5  majority vote tie-break: >= instead of > (D5)
mut M5 SRC 'if (votes[c] > votes[best]) best = c;' \
           'if (votes[c] >= votes[best]) best = c;'

# M6  majority vote: return COUNTS instead of one-hot (D6)
mut M6 SRC 'out(i, best) = 1.0;' 'out(i, best) = votes[best];'

# M7  hflip: flip the whole buffer instead of within each image row (D10)
mut M7 SRC 'out(r, i * width + (width - 1 - j)) = x(r, i * width + j);' \
           'out(r, i * width + (width - 1 - j)) = x(x.rows - 1 - r, i * width + j);'

# M8  weights: DON'T normalise (D8) -- {1,3} would no longer equal {2,6}
mut M8 SRC 'p.w[i] = weights[i] / total;' 'p.w[i] = weights[i];'

# M9  T == 1 guard removed in plan_aggregation: a single transform silently
#     passes through. The anchor is made UNIQUE by including the message, since
#     `require(bundle.rows > 1,` occurs twice (plan_aggregation and
#     majority_vote) and an unanchored replace would hit both, scoring as
#     "break both guards" -- which any T=1 fixture catches anyway and therefore
#     proves nothing about this one in isolation.
mut M9 SRC 'require(bundle.rows > 1,
            "tta: aggregation requires at least 2 transforms, got "' \
           'require(bundle.rows >= 1,
            "tta: aggregation requires at least 2 transforms, got "'

# M10 flip report: COUNT EVERY CHANGED ROW AS incorrect_to_correct, dropping the
#     base-was-right test (D9). Written to keep `base_right` USED so the mutant
#     compiles warning-free -- see the harness note below: a mutation that only
#     introduces an unused-variable warning is reported MUT-BROKEN, which is a
#     harness verdict about the BUILD, not evidence about the suite.
mut M10 SRC 'if (base_right) ++r.correct_to_incorrect;
        else            ++r.incorrect_to_correct;' \
            'if (base_right) ++r.incorrect_to_correct;
        (void)base_right; ++r.incorrect_to_correct;'

# M11 transform view: wrong index -- use the CLASS axis as the row axis (D1)
mut M11 SRC 'out(i, j) = bundle(transform, i * num_classes + j);' \
            'out(i, j) = bundle(transform, j * (bundle.cols / num_classes) + i);'

# M12 mean_probs: forget the softmax, average the raw logits. Written to consume
#     `probs` fully so the mutant is warning-free, and to keep the softmax helper
#     referenced so MUT-BROKEN cannot be confused with a caught mutation.
mut M12 SRC 'softmax_row_into(bundle, t, n, p.num_classes, probs);
            for (size_t c = 0; c < p.num_classes; ++c)
                out(n, c) += p.w[t] * probs(n, c);' \
            'for (size_t c = 0; c < p.num_classes; ++c)
                out(n, c) += p.w[t] * bundle(t, n * p.num_classes + c);
            if (p.num_rows == 0) softmax_row_into(bundle, t, n, p.num_classes, probs);'

echo
echo "HARNESS NOTE -- MUT-BROKEN IS NOT A VERDICT."
echo "Two mutations in an earlier draft of this file were reported MUT-BROKEN"
echo "and NEITHER of them was evidence about the suite:"
echo "  M10  introduced an unused-variable warning; the mutant was"
echo "        semantically VALID and the suite did catch it."
echo "  M12  left 'probs' written-but-never-read; also valid C++."
echo "The harness therefore refuses to score a mutant that merely warns,"
echo "which is correct -- but it means a mutation must be written to be"
echo "warning-free to produce a verdict at all. Both are rewritten above."
echo
echo "THIRD DEFECT, FOUND BY RUNNING THE BINARY AFTERWARDS: the harness"
echo "restores the SOURCE but leaves build/test_tta holding the LAST MUTANT."
echo "The stale binary then reports 73 passed / 4 failed and looks like a real"
echo "regression in the pristine tree. This is the stale-.o trap the"
echo "knowledge-distillation session recorded, arriving as a stale EXECUTABLE"
echo "instead. Fixed below: the harness rebuilds the real binary on exit."

# --- restore the real binary ------------------------------------------------
# Without this, build/test_tta is the LAST MUTANT, not the implementation.
# Verified by measurement: right after this note was written, the stale binary
# reported "73 passed, 4 failed" while a fresh build of the same pristine
# source reported "77 passed, 0 failed" -- i.e. it looked like a regression in
# code that had not changed at all.
restore
rm -f "$WORK"/*.o "$TESTBIN"
if make build/test_tta >/dev/null 2>&1 && "$TESTBIN" 2>&1 | tail -1 | grep -q '0 failed'; then
    echo
    echo "restored build/test_tta: pristine, $( "$TESTBIN" 2>&1 | tail -1 )"
else
    echo
    echo "HARNESS-DEFECT: could not rebuild a passing pristine binary."
fi

echo
echo "NOTE: a survivor is NOT a verdict. Per the TDD skill, an unexplained"
echo "survivor must be MEASURED (can it change anything?) before being called"
echo "inert, and the measurement gets its own inverted harness function."