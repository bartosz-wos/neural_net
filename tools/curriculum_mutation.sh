#!/usr/bin/env bash
# =============================================================================
# Mutation harness for tests/test_curriculum.cpp
#
# Every defect below was learned the hard way in this repo (nine recorded
# instances) and is defended against explicitly:
#
#  1. LINK SET. Two earlier harnesses parsed `LIB_OBJS := ...` out of the
#     Makefile with a regex. The first matched NOTHING (`:=` has a space before
#     it); the second matched but yielded ZERO .o files, because LIB_OBJS is a
#     RECURSIVE make variable whose literal text contains `$(LIB_SRCS:...)`.
#     Either bug scores every mutation `MUT-BROKEN`, i.e. it would "certify" a
#     suite that catches nothing. So: ask make, COUNT the objects, and ABORT
#     before reporting any verdict if the count is too low.
#
#  2. THE MUTATED OBJECT MUST NOT BE IN THE LINK SET, or every symbol is
#     double-defined and every mutation is falsely "caught" by a link error.
#     A link error is not a test failure.
#
#  3. EVERY EDIT IS VERIFIED WITH A CHECKSUM. A `sed`/`str.replace` that does not
#     apply is indistinguishable from a mutation that survived.
#
#  4. BOTH `.cpp` AND `.h` ARE SNAPSHOTTED AND RESTORED. A harness that
#     snapshots only the `.cpp` while mutating the header LEAKS the mutation into
#     the tree; the next run then reports N mutations failing identically, which
#     is the tell.
#
#  5. RESTORE IS ATOMIC: `cp base tmp && mv -f tmp src`. A plain `cp` truncates
#     the destination first, so a concurrent reader can splice two generations.
#
#  6. STALE OBJECTS. `rm -f` the intermediates before every build -- a leftover
#     .o from the previous mutation made one mutant report a build failure that
#     did not reproduce in isolation.
#
#  7. `MUT-BROKEN` (build failed) and `MUT-CRASH` (test aborted) are their OWN
#     columns. Neither is a mutation verdict. A mutation must compile
#     warning-free to produce a verdict at all.
#
#  8. REBUILD AND RE-VERIFY THE REAL BINARY ON EXIT. An earlier session
#     restored the source but left `build/test_tta` holding the LAST MUTANT,
#     which reported a phantom regression in code that had not changed.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
SRC_CPP=include/nn/utils/curriculum.cpp
SRC_H=include/nn/utils/curriculum.h
BIN=build/test_curriculum

echo "=== 0. harness self-checks ==="
[ -f "$SRC_CPP" ] || { echo "FATAL: missing $SRC_CPP"; exit 2; }
[ -f "$SRC_H" ]   || { echo "FATAL: missing $SRC_H"; exit 2; }

# --- defence 1: the link set comes from make, and is COUNTED ---------------
LIB_OBJS_RAW=$(make -s print-lib-objs 2>/dev/null | tr ' ' '\n' | grep '\.o$' | grep -v 'test_' | sort -u)
N_OBJS=$(printf '%s\n' "$LIB_OBJS_RAW" | grep -c . )
echo "link-set objects: $N_OBJS"
if [ "$N_OBJS" -lt 10 ]; then
    echo "FATAL: link set has $N_OBJS objects (< 10). Refusing to report any verdict."
    echo "       One harness defect must never be reported as N mutation verdicts."
    exit 2
fi

# --- defence 2: drop the mutated object from the link set ------------------
LIB_LINK=$(printf '%s\n' "$LIB_OBJS_RAW" | grep -v 'build/utils/curriculum\.o$' | tr '\n' ' ')
echo "linking $(printf '%s\n' $LIB_LINK | grep -c .) objects (curriculum.o excluded)"

BASE_CPP=$(mktemp) ; BASE_H=$(mktemp) ; TMPD=$(mktemp -d)
cp "$SRC_CPP" "$BASE_CPP"; cp "$SRC_H" "$BASE_H"

restore() {   # atomic
    cp "$BASE_CPP" "$TMPD/c" && mv -f "$TMPD/c" "$SRC_CPP"
    cp "$BASE_H"   "$TMPD/h" && mv -f "$TMPD/h" "$SRC_H"
}
trap 'restore' EXIT

CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -march=native -Iinclude"

build_and_run() {
    rm -f build/utils/curriculum.o "$BIN"          # defence 6
    if ! g++ $CXXFLAGS -c "$SRC_CPP" -o build/utils/curriculum.o 2>"$TMPD/cc.log"; then
        echo "BUILT-NO"; return 0
    fi
    # Warnings are checked ONLY in the MUTATED TU. The test TU includes the
    # whole umbrella header, which drags in a PRE-EXISTING
    # `-Wunused-parameter` in layers/utility/spatial_dropout.h -- so gating on
    # that log reported MUT-BROKEN for every single mutation, which is how two
    # perfectly valid mutants ended up scored as "build failures" when both
    # compile cleanly in isolation.
    if grep -q 'warning:' "$TMPD/cc.log"; then echo "BUILT-WARN"; return 0; fi
    if ! g++ $CXXFLAGS -c tests/test_curriculum.cpp -o "$TMPD/t.o" 2>"$TMPD/cc2.log"; then
        echo "BUILT-NO"; return 0
    fi
    # --- defence 2, ENFORCED: the mutated object must NOT be in the link set.
    #     A link error is NOT a mutation verdict: this harness's first run
    #     included build/utils/curriculum.o in $LIB_LINK as well as passing it
    #     explicitly, every link failed with `multiple definition`, and all 25
    #     mutations were scored SURVIVED. That is the repo's ninth instance of
    #     "one harness defect reported as N mutation verdicts", and it would
    #     have certified a suite that catches nothing.
    if printf '%s\n' $LIB_LINK | grep -q 'build/utils/curriculum\.o$'; then
        echo "LINK-DOUBLE"; return 0
    fi
    if ! g++ "$TMPD/t.o" $LIB_LINK build/utils/curriculum.o -o "$BIN" 2>"$TMPD/ld.log"; then
        # A link failure is MUT-BROKEN, never a survivor and never a catch.
        echo "BUILT-NO"; return 0
    fi
    OUT=$("$BIN" 2>&1); RC=$?
    if [ $RC -ge 128 ]; then echo "MUT-CRASH"; return 0; fi
    if echo "$OUT" | grep -q '=== Summary'; then
        echo "$OUT" | grep '=== Summary' | sed 's/.*Summary: //'
    else
        echo "MUT-CRASH"
    fi
    return 0
}

echo
echo "=== 1. pristine suite must be GREEN before any mutation ==="
PRISTINE=$(build_and_run); echo "pristine: $PRISTINE"
if ! echo "$PRISTINE" | grep -q ', 0 failed'; then
    echo "FATAL: pristine suite is not green ($PRISTINE). Refusing to run."
    exit 2
fi

# --- the mutations: tag | file | old | new ---------------------------------
MUTS=(
"Hard uses <= instead of <cppreturn (ell < lambda) ? 1.0 : 0.0;return (ell <= lambda) ? 1.0 : 0.0;"
"Hard always returns 1cppreturn (ell < lambda) ? 1.0 : 0.0;return 1.0;"
"Linear drops the clamp to 0cppreturn (ell < lambda) ? (1.0 - ell / lambda) : 0.0;return 1.0 - ell / lambda;"
"Linear sign flipcppreturn (ell < lambda) ? (1.0 - ell / lambda) : 0.0;return (ell < lambda) ? (ell / lambda) : 0.0;"
"Log swaps numerator/denominatorcppreturn std::log(ell + zeta) / std::log(zeta);return std::log(zeta) / std::log(ell + zeta);"
"Mixture threshold order swappedcppif (ell <= l2) return 1.0;if (ell >= l2) return 1.0;"
"Mixture drops the zero branchcppif (ell >= l1) return 0.0;if (ell >= l1) return 1.0;"
"Gamma cut threshold squared wronglycppconst double cut2 = cut * cut;const double cut2 = cut;"
"Gamma loses the 1/lambda termcppreturn gamma * (1.0 / std::sqrt(ell) - 1.0 / lambda);return gamma * (1.0 / std::sqrt(ell));"
"Cauchy loses the factor 2cppreturn 2.0 / (1.0 + (ell / lambda) * (ell / lambda));return 1.0 / (1.0 + (ell / lambda) * (ell / lambda));"
"Welsch loses the factor 2cppreturn 2.0 * std::exp(-(ell / lambda) * (ell / lambda));return std::exp(-(ell / lambda) * (ell / lambda));"
"L1L2 uses lam instead of lam+t^2cppreturn 1.0 / std::sqrt(lambda + ell * ell);return 1.0 / std::sqrt(lambda + 1.0);"
"L1L2 returns t instead of 1/sqrtcppreturn 1.0 / std::sqrt(lambda + ell * ell);return ell / std::sqrt(lambda + ell * ell);"
"Sec32 numerator becomes 1+tcppreturn 1.0 / (1.0 + ell / (lambda * lambda));return (1.0 + ell) / (1.0 + ell / (lambda * lambda));"
"Sec32 squares the wrong thingcppreturn 1.0 / (1.0 + ell / (lambda * lambda));return 1.0 / (1.0 + ell / lambda);"
"schedule multiply becomes dividecpp    return lambda * step_factor;    return lambda / step_factor;"
"schedule drops its monotonic guardcpp    if (!(step_factor > 1.0)) bad(\"multiplying requires a step factor > 1\");    if (false) bad(\"nope\");"
"lambda_init uses ceil not floorcppconst size_t lo = static_cast<size_t>(std::floor(pos));const size_t lo = static_cast<size_t>(std::ceil(pos));"
"lambda_init interpolates backwardscppdouble lam = sorted[lo] + frac * (sorted[hi] - sorted[lo]);double lam = sorted[hi] - frac * (sorted[hi] - sorted[lo]);"
"weighted_loss drops the weightscpptotal += weights(0, i) * losses(0, i);total += losses(0, i);"
"gradient normalises by Ncpp    for (size_t i = 0; i < losses.cols; ++i) g(0, i) = losses(0, i);    for (size_t i = 0; i < losses.cols; ++i) g(0, i) = losses(0, i) / static_cast<double>(losses.cols);"
"hard_fraction flips <= to >cppif (weights(0, i) <= tol) ++excluded;if (weights(0, i) > tol) ++excluded;"
"effective_mass stops after the firstcppfor (size_t i = 0; i < weights.cols; ++i) total += weights(0, i);for (size_t i = 0; i < 1; ++i) total += weights(0, i);"
"hq_minimiser returns a different quantitycpp    return curr_weight(mode, lambda, ell);    return curr_weight(mode, lambda, ell) * ell;"
"hq_minimiser stops refusing explicit modescpp        bad(\"this mode is an EXPLICIT regulariser and has no Table-1 counterpart\");        return curr_weight(mode, lambda, ell);"
"lambda guard weakened to a NaN-only checkcpp    if (!(lambda > 0.0) || std::isinf(lambda) || std::isnan(lambda))    if (std::isnan(lambda))"
"ell guard weakened to a NaN-only checkcpp    if (!(ell >= 0.0) || std::isnan(ell))    if (std::isnan(ell))"
)

echo
echo "=== 2. mutation runs ==="
CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0; WARN=0
FAILED_MUTS=""

for spec in "${MUTS[@]}"; do
    IFS=$'\x1f' read -r tag file old new <<< "$spec"
    restore
    target=$SRC_CPP; [ "$file" = "h" ] && target=$SRC_H
    # --- defence 3: apply with python str.replace (no sed metacharacters) ---
    python3 - "$target" "$old" "$new" <<'PYEOF'
import sys, hashlib
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
if src.count(old) != 1:
    sys.exit(9)          # anchor not unique -> harness defect, NOT a verdict
open(path, 'w').write(src.replace(old, new, 1))
PYEOF
    if [ $? -ne 0 ]; then
        printf '  %-46s MUT-NOOP   (anchor not unique/not found)\n' "$tag"
        NOOP=$((NOOP+1)); FAILED_MUTS="$FAILED_MUTS\n  $tag (NOOP)"
        continue
    fi
    # verify the edit actually landed, by content
    grep -qF -- "$new" "$target" || {
        printf '  %-46s MUT-NOOP   (edit did not apply)\n' "$tag"
        NOOP=$((NOOP+1)); FAILED_MUTS="$FAILED_MUTS\n  $tag (NOOP)"; continue; }

    RES=$(build_and_run); RC=$?
    case "$RES" in
        BUILT-NO)    printf '  %-46s MUT-BROKEN  (build/link failed -- NOT a verdict)\n' "$tag"
                     echo "         --- compiler/linker output ---"
                     sed 's/^/         /' "$TMPD/cc.log" | head -6
                     sed 's/^/         /' "$TMPD/cc2.log" | head -6
                     sed 's/^/         /' "$TMPD/ld.log" 2>/dev/null | head -6
                     BROKEN=$((BROKEN+1)); FAILED_MUTS="$FAILED_MUTS\n  $tag (BROKEN)";;
        LINK-DOUBLE) printf '  %-46s LINK-DOUBLE  (harness defect: mutant in link set)\n' "$tag"
                     BROKEN=$((BROKEN+1)); FAILED_MUTS="$FAILED_MUTS\n  $tag (LINK-DOUBLE)";;
        BUILT-WARN)  printf '  %-46s MUT-WARN   (compiled with warnings)\n' "$tag"
                     WARN=$((WARN+1)); FAILED_MUTS="$FAILED_MUTS\n  $tag (WARN)";;
        MUT-CRASH)   printf '  %-46s MUT-CRASH  (test aborted)\n' "$tag"
                     CRASH=$((CRASH+1)); FAILED_MUTS="$FAILED_MUTS\n  $tag (CRASH)";;
        *)
            # The summary line is "N passed, M failed ===" and `sed` strips only
            # the prefix, so NF arrives as "M ===". The first version compared
            # that with [ -gt 0 ] -- a SYNTACTIC ERROR in bash, which sends the
            # test to the `else` branch -- so EVERY mutation that the suite DID
            # catch was reported SURVIVED. 25 mutations, 0 real survivors, and
            # the harness was confidently wrong about every one of them.
            # Fingerprint to remember: a harness that reports 100% survival
            # against a suite you have watched catch real bugs is lying.
            NF=$(echo "$RES" | sed 's/.*passed, //; s/ failed.*//; s/[^0-9].*//')
            if [ -n "$NF" ] && [ "$NF" -gt 0 ] 2>/dev/null; then
                printf '  %-46s CAUGHT (%s assertions)\n' "$tag" "$NF"
                CAUGHT=$((CAUGHT+1))
            else
                printf '  %-46s SURVIVED   <-- suite does not catch this\n' "$tag"
                SURVIVED=$((SURVIVED+1)); FAILED_MUTS="$FAILED_MUTS\n  $tag (SURVIVED)"
            fi;;
    esac
done

restore
echo
echo "=== 3. summary ==="
echo "caught=$CAUGHT  survived=$SURVIVED  broken=$BROKEN  crash=$CRASH  noop=$NOOP  warn=$WARN"
[ -n "$FAILED_MUTS" ] && { echo "needs attention:"; echo -e "$FAILED_MUTS"; }

# --- defence 8: rebuild and re-verify the REAL binary on exit --------------
echo
echo "=== 4. post-run verification (rebuild from restored source) ==="
FINAL=$(build_and_run)
echo "restored suite: $FINAL"
restore
rm -f "$BASE_CPP" "$BASE_H"; rm -rf "$TMPD"
echo "$FINAL" | grep -q ', 0 failed' && { echo "TREE CLEAN + SUITE GREEN"; exit 0; }
echo "TREE OR SUITE NOT CLEAN -- DO NOT TRUST THIS RUN"; exit 1
