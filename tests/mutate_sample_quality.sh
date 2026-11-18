#!/usr/bin/env bash
# =============================================================================
# tests/mutate_sample_quality.sh -- mutation harness for the sample-quality
# metrics: utils/sample_quality.{h,cpp}.
#
# WHY THIS FILE EXISTS
# =============================================================================
#
# A suite is, by construction, unable to report a coverage gap: it can only
# report what it asserts. This harness breaks the implementation in specific
# ways and requires that the suite FAIL each time. It is the only thing in
# this repo that can find "the suite is green and the behaviour is wrong".
#
# =============================================================================
# THE MUTATIONS, AND WHAT EACH ONE IS FOR
# =============================================================================
#
#   M1  the FID returns the DISTANCE d instead of the SQUARED distance d^2
#       -> the squaring convention (C1). THE most common FID error in the
#          wild, and the shipped `frechet_distance` returns the distance,
#          so this is the mutation the implementation is most likely to
#          ship by accident.
#   M2  the covariance uses the BIASED estimator (divide by n)
#       -> the unbiased-estimator convention (C2).
#   M3  the covariance's normaliser is n+1
#       -> the normaliser, from the other side.
#   M4  the covariance's OFF-DIAGONAL is dropped (a diagonal covariance)
#       -> that the FULL covariance is used.
#   M5  the covariance's symmetrising assignment is dropped (only the
#       upper triangle is written)
#       -> the symmetry.
#   M6  the mean term is ||dmu|| instead of ||dmu||^2
#       -> the mean term's weight.
#   M7  the IS's zero guard is removed (the `p <= 0` skip)
#       -> the 0*log(0/q) = 0 convention (C5). Without it the IS is NaN on
#          a one-hot fixture.
#   M8  the IS's marginal is computed over the WHOLE set instead of over
#       the split's own rows
#       -> the split semantics.
#   M9  the IS's split count is ignored (one split covering everything)
#       -> the split convention (C6).
#   M10 the exp() is outside the group mean
#       -> the exp()'s placement.
#   M11 the exponentiation is dropped entirely
#       -> that the IS is an exp at all.
#   M12 the feature-dimension guard is dropped
#       -> the guard that turns a shape mismatch into an ERROR rather than
#          a wrong number.
#   M13 the n>=2 guard is dropped
#       -> the precondition the unbiased estimator needs.
#   M14 the row-sum guard is dropped
#       -> the guard that makes the metric a divergence between
#          DISTRIBUTIONS.
#   M15 the negative-probability guard is dropped
#       -> that guard.
#   M16 the covariance's MEAN is not subtracted (the centred outer sum
#       becomes E[xx^T])
#       -> the numerically stable centred form. **This one is a
#          CATASTROPHIC-CANCELLATION mutation**: the two forms are
#          algebraically equal, so a test on random data cannot separate
#          them, and the difference only appears when the mean is large
#          relative to the spread.
#   M17 the FID's sample Gaussians are swapped
#       -> **MEASURED INERT**: FID is mathematically symmetric in its two
#          arguments (S1.S2 and S2.S1 have the same eigenvalues, so the two
#          trace terms are equal). The harness scores it with an INVERTED
#          verdict: it PASSES while the mutation is inert and FAILS when it
#          starts to matter. The probe is tests/sample_quality_inertness.py
#          -- the SAME probe the oracle's harness uses, because the
#          property is a property of the MATHEMATICS and not of the
#          language.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY PAID FOR
# =============================================================================
#
#   1. `restore()` must NEVER delete its backup. It is called once PER
#      MUTATION and once on EXIT, so a self-cleaning restore works exactly
#      once and every later restore is a silent no-op -- leaving a MUTANT IN
#      THE TREE. Cleanup happens exactly once, on EXIT, and restore() is
#      ATOMIC (`cp tmp && mv -f`), because a plain `cp` truncates the
#      destination first, so a concurrent reader can splice two generations
#      together.
#   2. The link set comes from `make print-lib-objs`, NEVER from
#      regex-parsing the Makefile. `LIB_OBJS` is a RECURSIVE make variable
#      whose literal text contains no `.o` filenames, so a regex yields an
#      empty set and every mutation scores MUT-BROKEN -- one defect
#      reported as N verdicts. The count is asserted >= 10.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET. It legitimately
#      IS in the library's list (it is the TU being mutated), so the
#      correct handling is to compile the mutated TU SEPARATELY, drop the
#      library's copy from the link set, and link the mutant in its place.
#      Excluding it without replacing it breaks the link for the wrong
#      reason.
#   4. `MUT-BROKEN` means "the build failed". It is a separate class and is
#      never counted as a pass. A mutation reported BROKEN is re-run in
#      isolation before it is believed.
#   5. A CRASH (rc >= 128) is a THIRD class, not a verdict.
#   6. A mutation that DID NOT APPLY is a HARNESS DEFECT, not a
#      "SURVIVED" verdict. The tree is checksummed before and after every
#      edit.
#   7. The anchor must occur EXACTLY ONCE, and the count comes from the
#      SAME engine that performs the edit (`str.count` in Python), because
#      `grep -c` counts MATCHING LINES, not matches.
#   8. `rm -f` the intermediates before every build -- a stale .o from the
#      previous mutation reproduces a build failure that does not exist
#      alone.
#   9. NEVER run a build concurrently with this harness.
#  10. The tree is rebuilt and re-verified ON EXIT before any verdict is
#      reported.
#  11. The pristine tree must be GREEN before any mutation runs. A harness
#      that starts from a red tree reports garbage.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 99

SRC_CPP="include/nn/utils/sample_quality.cpp"
SRC_HDR="include/nn/utils/sample_quality.h"
MUT_OBJ="build/utils/sample_quality.o"
SUITE="build/test_sample_quality"

BAK_CPP="$(mktemp)"; BAK_HDR="$(mktemp)"
cp "$SRC_CPP" "$BAK_CPP"
cp "$SRC_HDR" "$BAK_HDR"

SCRATCH="$(mktemp -d)"

restore() {
    cp "$BAK_CPP" "$SRC_CPP.tmp" 2>/dev/null && mv -f "$SRC_CPP.tmp" "$SRC_CPP"
    cp "$BAK_HDR" "$SRC_HDR.tmp" 2>/dev/null && mv -f "$SRC_HDR.tmp" "$SRC_HDR"
}

cleanup() {
    restore
    rm -f "$BAK_CPP" "$BAK_HDR"
    rm -rf "$SCRATCH"
    # THE POST-HARNESS REBUILD-AND-VERIFY DEFENCE. A previous session shipped
    # a restored source with a stale mutant still on disk and reported a
    # phantom regression; this is the check that catches it.
    rm -f "$MUT_OBJ" build/test_sample_quality.o "$SUITE"
    make setup >/dev/null 2>&1
    if make -j6 "$SUITE" >/dev/null 2>&1; then
        if "$SUITE" >/dev/null 2>&1; then
            echo ""
            echo "POST-HARNESS: tree restored, rebuilt, and GREEN."
        else
            echo ""
            echo "POST-HARNESS FAILURE: the restored tree does not pass. STOP." >&2
            exit 1
        fi
    else
        echo ""
        echo "POST-HARNESS FAILURE: the restored tree does not BUILD. STOP." >&2
        exit 1
    fi
}
trap cleanup EXIT INT TERM

checksum() { md5sum "$1" | cut -d' ' -f1; }

# -----------------------------------------------------------------------------
# The link set, from make itself, with a count assertion.
# -----------------------------------------------------------------------------
mapfile -t ALL_OBJS < <(make print-lib-objs)
if [ "${#ALL_OBJS[@]}" -lt 10 ]; then
    echo "HARNESS DEFECT: link set has only ${#ALL_OBJS[@]} objects (need >= 10)." >&2
    echo "REFUSING TO RUN -- every verdict below would be false." >&2
    exit 99
fi
LIB_LINK=""
for o in "${ALL_OBJS[@]}"; do
    case "$o" in
        *sample_quality.o|*test_sample_quality.o) ;;
        *) LIB_LINK="$LIB_LINK $o" ;;
    esac
done
LIB_COUNT=$(echo "$LIB_LINK" | wc -w)
if [ "$LIB_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: link set collapsed to $LIB_COUNT objects." >&2
    exit 99
fi
echo "link set: $LIB_COUNT objects (the mutated TU is compiled separately and linked in)"

CXXFLAGS_LOCAL="-std=c++17 -O2 -Wall -Wextra -march=native -Iinclude"

# -----------------------------------------------------------------------------
# The test translation unit is compiled ONCE, up front: it depends on
# sample_quality.h, which no mutation touches (every mutation is in the
# .cpp), so recompiling it per mutation would multiply the harness's
# runtime for no extra coverage. The mutated TU IS recompiled every time.
# -----------------------------------------------------------------------------
prepare_tests() {
    g++ $CXXFLAGS_LOCAL -c "tests/test_sample_quality.cpp" -o "$SCRATCH/test_sample_quality.o" 2>"$SCRATCH/build.log" || return 90
    return 0
}

build_suites() {
    rm -f "$MUT_OBJ" "$SCRATCH/sample_quality.o" "$SCRATCH/test_sample_quality"
    g++ $CXXFLAGS_LOCAL -c "$SRC_CPP" -o "$SCRATCH/sample_quality.o" 2>"$SCRATCH/build.log" || return 90
    g++ $CXXFLAGS_LOCAL "$SCRATCH/test_sample_quality.o" "$SCRATCH/sample_quality.o" $LIB_LINK \
        -o "$SCRATCH/test_sample_quality" 2>>"$SCRATCH/build.log" || return 90
    return 0
}

run_suite() {  # $1 = suite name
    local out rc
    out=$("$SCRATCH/$1" 2>&1); rc=$?
    LAST_OUT="$out"
    return $rc
}

failed_count() {
    echo "$1" | sed -nE 's/.*SUMMARY: ([0-9]+) passed, ([0-9]+) failed.*/\2/p' | tail -1
}

passed_count() {
    echo "$1" | sed -nE 's/.*SUMMARY: ([0-9]+) passed, ([0-9]+) failed.*/\1/p' | tail -1
}

# -----------------------------------------------------------------------------
# The pristine self-check: the suite must be GREEN before any mutation runs.
# -----------------------------------------------------------------------------
prepare_tests
if [ $? -ne 0 ]; then
    echo "HARNESS DEFECT: the test translation unit does not compile." >&2
    tail -20 "$SCRATCH/build.log" >&2
    exit 99
fi
build_suites
if [ $? -ne 0 ]; then
    echo "HARNESS DEFECT: the PRISTINE tree does not build." >&2
    tail -20 "$SCRATCH/build.log" >&2
    exit 99
fi
PRISTINE_OK=1
run_suite test_sample_quality || { echo "HARNESS DEFECT: pristine suite exits non-zero." >&2; PRISTINE_OK=0; }
nf=$(failed_count "$LAST_OUT")
if [ -z "$nf" ] || [ "$nf" -ne 0 ]; then
    echo "HARNESS DEFECT: pristine suite is not green (${nf:-no summary} failed)." >&2
    PRISTINE_OK=0
else
    echo "pristine: test_sample_quality -- $(passed_count "$LAST_OUT") passed, 0 failed"
fi
[ "$PRISTINE_OK" -eq 1 ] || exit 99
echo "pristine: green -- mutation may begin"

# -----------------------------------------------------------------------------
# Apply a mutation with a Python editor (str.count / str.replace). perl's
# s/// dies on C++ anchors containing '/', and `grep -c` counts LINES.
# The spec is "old\x1fnew" -- 0x1f cannot occur in either side.
# -----------------------------------------------------------------------------
apply_mutation() {  # $1 = spec (old<US>new), $2 = file
    python3 - "$1" "$2" <<'PYEOF'
import sys, pathlib
spec, path = sys.argv[1], sys.argv[2]
old, new = spec.split("\x1f")
p = pathlib.Path(path)
s = p.read_text()
n = s.count(old)
if n != 1:
    print(f"ANCHOR_COUNT={n}")
    sys.exit(3)
p.write_text(s.replace(old, new))
print("MUTATION_APPLIED")
PYEOF
}

# THE SEPARATOR. 0x1f (the "unit separator") cannot occur in either side of
# a mutation spec, so it is safe as a delimiter -- unlike '/', which C++ is
# FULL of and which killed a previous harness's perl s/// script.
#
# **IT MUST BE A REAL BYTE.** `$'\x1f'` produces it; `"\\x1f"` inside
# double quotes produces the four literal characters, and Python's
# `split("\x1f")` then finds no separator and raises ValueError -- which
# is what the FIRST DRAFT of a sibling harness did, and every mutation was
# reported HARNESS-DEFECT while the verdict table still looked plausible.
US=$'\x1f'

CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0; INERT=0
declare -a TABLE

print_table() {
    printf '%-6s %-52s %-42s %s\n' "ID" "MUTATION" "VERDICT" "DETAIL"
    printf '%-6s %-52s %-42s %s\n' "----" "----------------------------------------------------" "------------------------------------------" "------"
    local row a b c d
    for row in "${TABLE[@]}"; do
        IFS='|' read -r a b c d <<< "$row"
        printf '%-6s %-52s %-42s %s\n' "$a" "$b" "$c" "${d:-}"
    done
}

# -----------------------------------------------------------------------------
# run_mutation <id> <description> <old> <new>
# -----------------------------------------------------------------------------
run_mutation() {
    local id="$1" desc="$2" old="$3" new="$4"
    restore

    local out
    out=$(apply_mutation "$old${US}$new" "$SRC_CPP")
    if ! echo "$out" | grep -q "MUTATION_APPLIED"; then
        TABLE+=("$id|$desc|HARNESS-DEFECT (anchor count != 1)|$(echo "$out" | tr '\n' ' ')")
        NOOP=$((NOOP+1)); return
    fi

    # Verify the edit really landed.
    local before after
    before=$(checksum "$BAK_CPP"); after=$(checksum "$SRC_CPP")
    if [ "$before" = "$after" ]; then
        restore
        TABLE+=("$id|$desc|HARNESS-DEFECT (edit did not land)|")
        NOOP=$((NOOP+1)); return
    fi

    build_suites
    local rc=$?
    if [ "$rc" -eq 90 ]; then
        restore
        TABLE+=("$id|$desc|MUT-BROKEN (build failed -- NOT a verdict)|")
        BROKEN=$((BROKEN+1)); return
    fi

    run_suite test_sample_quality; rc=$?
    local suite_out="$LAST_OUT"
    local nf
    nf=$(failed_count "$suite_out")
    if [ "$rc" -ge 128 ]; then
        restore
        TABLE+=("$id|$desc|MUT-CRASH (rc=$rc -- NOT a verdict)|")
        CRASH=$((CRASH+1)); return
    fi
    if [ -n "$nf" ] && [ "$nf" -gt 0 ]; then
        local line
        line=$(echo "$suite_out" | grep -m1 "FAIL" | sed 's/^ *//' | cut -c1-70)
        TABLE+=("$id|$desc|CAUGHT|$line")
        CAUGHT=$((CAUGHT+1))
    else
        TABLE+=("$id|$desc|SURVIVED (the suite stayed green)|")
        SURVIVED=$((SURVIVED+1))
    fi
    restore
}

echo
echo "=== sample_quality mutations ==="

# --- M1: the FID returns d instead of d^2 -----------------------------------
run_mutation M1 "FID returns the distance d instead of d^2" \
    "    return d * d;" \
    "    return d;"

# --- M2: the biased covariance estimator ------------------------------------
run_mutation M2 "the covariance uses the BIASED estimator (divide by n)" \
    "    const double denom = static_cast<double>(n - 1);" \
    "    const double denom = static_cast<double>(n);"

# --- M3: the wrong normaliser ------------------------------------------------
run_mutation M3 "the covariance divides by n+1" \
    "    const double denom = static_cast<double>(n - 1);" \
    "    const double denom = static_cast<double>(n + 1);"

# --- M4: the covariance's off-diagonal is dropped ----------------------------
run_mutation M4 "the covariance's OFF-DIAGONAL is dropped" \
    "            covariance(j, k) = v;
            covariance(k, j) = v;   // symmetric by construction" \
    "            covariance(j, k) = v;
            if (j != k) continue;"

# --- M5: the covariance's symmetry is dropped -------------------------------
run_mutation M5 "the covariance's lower triangle is left unwritten" \
    "            covariance(j, k) = v;
            covariance(k, j) = v;   // symmetric by construction" \
    "            covariance(j, k) = v;"

# --- M6: THE SAMPLE MEAN'S NORMALISER --------------------------------------
# THE FIRST DRAFT OF THIS MUTATION TARGETED THE MEAN TERM'S SQUARING
# (`||dmu||^2` vs `||dmu||`), and it was a HARNESS DEFECT: that code lives
# in `utils/linear_algebra.cpp` -- inside the SHIPPED `frechet_distance` --
# and NOT in this file. The old and new strings were therefore IDENTICAL
# here, the edit could not land, and the harness reported
# "HARNESS-DEFECT (edit did not land)".
#
# **A MUTATION THAT TARGETS ANOTHER FILE'S CODE IS NOT THIS HARNESS'S
# BUSINESS**, and the repo has a standing rule for exactly this: the
# decision belongs to the harness that owns the file
# (`tests/mutate_linear_algebra.sh`). The mean term's squaring is pinned
# in the C++ suite by the SCALE-EQUIVARIANCE check (2.9 -- a mean term of
# `||dmu||` would scale by c rather than c^2), and on the specification
# side by the oracle harness's SM4.
#
# This mutation therefore targets a decision that IS this file's: the
# sample MEAN's normaliser. The mean is computed HERE, and dividing by
# the wrong count is a real defect this file can ship.
run_mutation M6 "the sample mean's normaliser is wrong (no division by n)" \
    "        mean(0, j) = sum / static_cast<double>(n);" \
    "        mean(0, j) = sum;"

# --- M7: the IS's zero guard -----------------------------------------------
run_mutation M7 "the IS's zero guard is removed" \
    "                if (p <= 0.0) continue;" \
    "                if (p < 0.0) continue;"

# --- M8: the marginal is over the whole set --------------------------------
run_mutation M8 "the IS's marginal is over the WHOLE set" \
    "            marginal(0, c) = sum / static_cast<double>(rows);" \
    "            marginal(0, c) = sum / static_cast<double>(n);"

# --- M9: the split count is ignored ----------------------------------------
run_mutation M9 "the IS's split count is ignored (one split)" \
    "        const size_t lo = static_cast<size_t>(s) * n / static_cast<size_t>(splits);
        const size_t hi = static_cast<size_t>(s + 1) * n / static_cast<size_t>(splits);" \
    "        const size_t lo = 0;
        const size_t hi = n;"

# --- M10: the exp() is outside the group mean ------------------------------
run_mutation M10 "the exp() is outside the group mean" \
    "        total += std::exp(kl_sum / static_cast<double>(rows));" \
    "        total += kl_sum / static_cast<double>(rows);"

# --- M11: the exponentiation is dropped ------------------------------------
run_mutation M11 "the IS's exponentiation is dropped" \
    "        total += std::exp(kl_sum / static_cast<double>(rows));" \
    "        total += kl_sum / static_cast<double>(rows) + 1.0;"

# --- M12: the feature-dimension guard --------------------------------------
run_mutation M12 "the feature-dimension guard is dropped" \
    "    if (real_features.cols != fake_features.cols)" \
    "    if (false)"

# --- M13: the n >= 2 guard --------------------------------------------------
run_mutation M13 "the n>=2 guard is dropped" \
    "    if (features.rows < 2)" \
    "    if (false)"

# --- M14: the row-sum guard -------------------------------------------------
run_mutation M14 "the IS's row-sum guard is dropped" \
    "        if (std::fabs(row_sum - 1.0) > 1e-6)" \
    "        if (false)"

# --- M15: the negative-probability guard -----------------------------------
run_mutation M15 "the IS's negative-probability guard is dropped" \
    "            if (p < 0.0)" \
    "            if (false)"

# --- M16: the covariance is NOT centred (catastrophic cancellation) ---------
run_mutation M16 "the covariance is not centred (E[xx^T] instead)" \
    "                const double dj = features(i, j) - mean(0, j);
                const double dk = features(i, k) - mean(0, k);" \
    "                const double dj = features(i, j);
                const double dk = features(i, k);"

# --- M17: the FID's sample Gaussians are swapped (MEASURED INERT) ----------
run_mutation M17 "the FID's sample Gaussians are swapped (PROVABLY INERT)" \
    "    const GaussianParams real = sample_gaussian_parameters(real_features);
    const GaussianParams fake = sample_gaussian_parameters(fake_features);" \
    "    const GaussianParams real = sample_gaussian_parameters(fake_features);
    const GaussianParams fake = sample_gaussian_parameters(real_features);"

echo
echo "=== caught $CAUGHT / survived $SURVIVED / broken $BROKEN / crashed $CRASH / noop $NOOP ==="

# -----------------------------------------------------------------------------
# THE INVERTED VERDICT FOR THE PROVABLY-INERT MUTATION (M17).
#
# M17 SURVIVES the suite, and that is a CORRECT verdict rather than a
# coverage gap: FID is MATHEMATICALLY SYMMETRIC in its two arguments.
# S1^(1/2).S2.S1^(1/2) is similar to S1.S2 and S2^(1/2).S1.S2^(1/2) is
# similar to S2.S1, and S1.S2 and S2.S1 have the SAME eigenvalues, so the
# two trace terms are equal and so are the two radicands.
#
# **A SURVIVOR MUST NEVER BE SILENTLY DROPPED**, and an "inert" verdict
# must itself be CHECKABLE -- otherwise a future refactor that breaks the
# symmetry would leave the harness reporting a stale "inert" that is no
# longer true. The verdict is therefore INVERTED: it PASSES while the
# mutation is inert and FAILS when it starts to matter. The probe is
# tests/sample_quality_inertness.py -- the SAME probe the oracle's harness
# uses, because the property is a property of the MATHEMATICS.
#
# **THE INVERTED VERDICT MAY ONLY DOWNGRADE A VERDICT THAT WAS ACTUALLY
# "SURVIVED".** A verdict about a mutation that did not run is not a
# verdict at all.
# -----------------------------------------------------------------------------
echo ""
echo "=== INVERTED VERDICT: is M17 still provably inert? ==="
M17_VERDICT=""
for row in "${TABLE[@]}"; do
    case "$row" in
        "M17|"*) M17_VERDICT="$row" ;;
    esac
done
case "$M17_VERDICT" in
    *"SURVIVED"*) ;;
    *)
        echo "M17  NOT-APPLICABLE  (M17's verdict was not SURVIVED -- the inertness claim cannot be scored)"
        echo "      M17's actual verdict: ${M17_VERDICT:-<missing>}"
        NOOP=$((NOOP+1))
        print_table
        echo ""
        echo "FINAL: $CAUGHT caught / $SURVIVED survived / $INERT measured-inert / $BROKEN broken / $CRASH crashed / $NOOP noop"
        exit 1
        ;;
esac
if /usr/bin/python3 tests/sample_quality_inertness.py > "$SCRATCH/inert.txt" 2>&1; then
    echo "M17  MEASURED-INERT  (the inverted-verdict check PASSES: the claim still holds)"
    grep -E "worst relative diff|BIT-IDENTICAL" "$SCRATCH/inert.txt" | sed 's/^/      /'
    SURVIVED=$((SURVIVED-1)); INERT=$((INERT+1))
    declare -a NEW_TABLE=()
    local_new=""
    for row in "${TABLE[@]}"; do
        case "$row" in
            "M17|"*) local_new="M17|the FID's sample Gaussians are swapped|MEASURED-INERT (the inverted verdict confirms it)|" ;;
            *) local_new="$row" ;;
        esac
        NEW_TABLE+=("$local_new")
    done
    TABLE=("${NEW_TABLE[@]}")
else
    echo "M17  NOT-INERT  (the inverted-verdict check FAILED -- the 'inert' verdict is now WRONG)"
    grep -E "\[FAIL\]" "$SCRATCH/inert.txt" | sed 's/^/      /'
fi

echo ""
print_table
echo ""
echo "FINAL: $CAUGHT caught / $SURVIVED survived / $INERT measured-inert / $BROKEN broken / $CRASH crashed / $NOOP noop"

[ "$SURVIVED" -eq 0 ] || { echo "NOTE: $SURVIVED unexplained survivor(s) -- these are REAL coverage gaps." >&2; exit 3; }
[ "$BROKEN" -eq 0 ] || { echo "NOTE: $BROKEN mutation(s) reported BROKEN/CRASH -- re-run in isolation." >&2; exit 1; }
[ "$NOOP" -eq 0 ] || { echo "NOTE: $NOOP harness defect(s)." >&2; exit 1; }
exit 0
