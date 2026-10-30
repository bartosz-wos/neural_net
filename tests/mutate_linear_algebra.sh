#!/usr/bin/env bash
# =============================================================================
# tests/mutate_linear_algebra.sh -- mutation harness for the PUBLIC symmetric
# linear algebra: utils/linear_algebra.{h,cpp}.
#
# WHY THIS FILE REPLACES tests/mutate_jacobi_eig.sh's TARGETS
# =============================================================================
#
# `tests/mutate_jacobi_eig.sh` mutated the rotation angle inside the three
# OPTIMIZERS' private copies. On 2026-10-29 those three copies were deleted and
# the routine was promoted to ONE public function
# (`nn::utils::symmetric_eigendecompose`), so the old harness's anchors --
# `theta = 0.5 * std::atan2(2.0 * apq, app - aqq);` inside shampoo.cpp, soap.cpp
# and galore.cpp -- NO LONGER EXIST anywhere.
#
# **THAT IS THE RECORDED HARNESS-DEFECT CLASS: "a mutation harness's ANCHORS
# are invalidated by adding a feature, and the failure is quiet."** The old
# harness would now report every mutation as MUT-NOOP/AMBIGUOUS, which reads
# like "the mutation is inert" when it actually means "the anchor is gone".
# The harness below re-anchors every mutation on the ONE file that now holds
# the routine, and it says so in its output.
#
# `tests/mutate_jacobi_eig.sh` is KEPT and still passes -- it measures the
# INERTNESS of the `app == aqq` degenerate branch, and `tests/test_jacobi_
# inertness.cpp` re-derives both angle variants INDEPENDENTLY of the library,
# so it does not care where the library keeps its copy. The mutant for that
# claim is applied HERE instead, to the shared file.
#
# =============================================================================
# THE MUTATIONS, AND WHAT EACH ONE IS FOR
# =============================================================================
#
# The routine has FOUR load-bearing decisions, and the suite has one suite per
# decision:
#
#   M1  the rotation angle's SIGN           -> tests/test_jacobi_eig.cpp
#   M2  the DESCENDING sort of the eigenpairs-> tests/test_linear_algebra.cpp
#                                              + tests/test_galore_projection.cpp
#   M3  the `max(lam, 0)` clamp             -> tests/test_linear_algebra.cpp
#   M4  the Frechet PRODUCT ORDER           -> tests/test_linear_algebra.cpp
#   M5  the EIGENVECTOR PERMUTATION (sort the eigenvalues but not the
#       eigenvectors -- the pairing bug)    -> tests/test_linear_algebra.cpp
#   M6  the input SYMMETRISATION             -> tests/test_linear_algebra.cpp
#   M7  the `app == aqq` degenerate branch   -> MEASURED INERT (see below)
#   M8  the guard on the mean dimension      -> tests/test_linear_algebra.cpp
#
# M7 IS MEASURED INERT, AND THAT IS A VERDICT IN ITS OWN RIGHT: with the
# corrected general formula, `atan2(2*apq, 0)` is EXACTLY +-pi/2, so the
# general branch already produces +-pi/4 -- bit-identically to what the
# special case hardcodes. `tests/test_jacobi_inertness.cpp` re-derives both
# variants independently and FAILS if that ever stops holding, so the claim
# is continuously verifiable rather than a one-off line in a harness log.
# The harness scores M7 with the INVERTED VERDICT: it PASSES when the
# mutation is still inert and FAILS when it starts to matter.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
# =============================================================================
#
#   1. `restore()` must NEVER delete its backup. It is called once PER
#      MUTATION and once on EXIT, so a self-cleaning restore works exactly
#      once and every later restore is a silent no-op -- leaving a MUTANT IN
#      THE TREE. Cleanup happens exactly once, on EXIT, and restore() is
#      ATOMIC (`cp tmp && mv -f`), because a plain `cp` truncates the
#      destination first and a concurrent reader can splice two generations
#      together.
#   2. The link set comes from `make print-lib-objs`, NEVER from
#      regex-parsing the Makefile. `LIB_OBJS` is a RECURSIVE make variable
#      whose literal text contains no `.o` filenames, so a regex yields an
#      empty set and every mutation scores MUT-BROKEN -- one defect reported
#      as N verdicts. The count is asserted >= 10.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET. It legitimately IS
#      in the library's list (it is the TU being mutated), so the correct
#      handling is to compile the mutated TU SEPARATELY, drop the library's
#      copy from the link set, and link the mutant in its place. Excluding it
#      without replacing it breaks the link for the wrong reason.
#   4. `MUT-BROKEN` means "the build failed". It is a separate class and is
#      never counted as a pass. A mutation reported BROKEN is re-run in
#      isolation before it is believed.
#   5. A CRASH (rc >= 128) is a THIRD class, not a verdict.
#   6. A mutation that DID NOT APPLY is a HARNESS DEFECT, not a "SURVIVED"
#      verdict. The tree is checksummed before and after every edit.
#   7. The anchor must occur EXACTLY ONCE, and the count comes from the SAME
#      engine that performs the edit (`str.count` in Python), because
#      `grep -c` counts MATCHING LINES, not matches.
#   8. `rm -f` the intermediates before every build -- a stale .o from the
#      previous mutation reproduces a build failure that does not exist alone.
#   9. NEVER run a build concurrently with this harness.
#  10. The tree is rebuilt and re-verified ON EXIT before any verdict is
#      reported.
#  11. The pristine tree must be GREEN before any mutation runs. A harness
#      that starts from a red tree reports garbage.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 99

SRC_CPP="include/nn/utils/linear_algebra.cpp"
SRC_HDR="include/nn/utils/linear_algebra.h"
MUT_OBJ="build/utils/linear_algebra.o"

# The suites that must catch a mutation of the shared routine. Each is
# linked against the MUTANT.
SUITE_LA="build/test_linear_algebra"     # the public API itself
SUITE_JE="build/test_jacobi_eig"         # the optimizers, through their PUBLIC accessors
SUITE_GP="build/test_galore_projection"  # GaLore's "top-r" contract

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
    rm -f "$MUT_OBJ" build/test_linear_algebra.o build/test_jacobi_eig.o \
          build/test_galore_projection.o "$SUITE_LA" "$SUITE_JE" "$SUITE_GP"
    make setup >/dev/null 2>&1
    if make -j6 "$SUITE_LA" "$SUITE_JE" "$SUITE_GP" >/dev/null 2>&1; then
        if "$SUITE_LA" >/dev/null 2>&1 && "$SUITE_JE" >/dev/null 2>&1 && "$SUITE_GP" >/dev/null 2>&1; then
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
        *linear_algebra.o|*test_linear_algebra.o|*test_jacobi_eig.o|*test_galore_projection.o) ;;
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
# Build the three suites against the CURRENT (possibly mutated) source.
# The mutated TU is compiled into $SCRATCH and linked IN PLACE of the
# library's copy. Returns 90 on a build failure -- which is NOT a verdict.
#
# The three TEST translation units are compiled ONCE, up front: they depend on
# linear_algebra.h, which no mutation touches (every mutation is in the .cpp),
# so recompiling them per mutation would multiply the harness's runtime by
# threefold for no extra coverage. The mutated TU IS recompiled every time.
# -----------------------------------------------------------------------------
prepare_tests() {
    local t
    for t in test_linear_algebra test_jacobi_eig test_galore_projection; do
        g++ $CXXFLAGS_LOCAL -c "tests/$t.cpp" -o "$SCRATCH/$t.o" 2>"$SCRATCH/build.log" || return 90
    done
    return 0
}

build_suites() {
    rm -f "$MUT_OBJ" "$SCRATCH/linear_algebra.o" "$SCRATCH"/test_linear_algebra \
          "$SCRATCH"/test_jacobi_eig "$SCRATCH"/test_galore_projection
    g++ $CXXFLAGS_LOCAL -c "$SRC_CPP" -o "$SCRATCH/linear_algebra.o" 2>"$SCRATCH/build.log" || return 90
    local t
    for t in test_linear_algebra test_jacobi_eig test_galore_projection; do
        g++ $CXXFLAGS_LOCAL "$SCRATCH/$t.o" "$SCRATCH/linear_algebra.o" $LIB_LINK \
            -o "$SCRATCH/$t" 2>>"$SCRATCH/build.log" || return 90
    done
    return 0
}

# Run one suite and echo its "N passed, M failed" counts. Returns the exit code.
run_suite() {  # $1 = suite name
    local out rc
    out=$("$SCRATCH/$1" 2>&1); rc=$?
    LAST_OUT="$out"
    return $rc
}
failed_count() {  # $1 = suite output
    echo "$1" | sed -nE 's/.*SUMMARY: ([0-9]+) passed, ([0-9]+) failed.*/\2/p' | tail -1
}
passed_count() {
    echo "$1" | sed -nE 's/.*SUMMARY: ([0-9]+) passed, ([0-9]+) failed.*/\1/p' | tail -1
}

# -----------------------------------------------------------------------------
# The pristine self-check: the suites must be GREEN before any mutation runs.
# -----------------------------------------------------------------------------
prepare_tests
if [ $? -ne 0 ]; then
    echo "HARNESS DEFECT: the test translation units do not compile." >&2
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
for t in test_linear_algebra test_jacobi_eig test_galore_projection; do
    run_suite "$t" || { echo "HARNESS DEFECT: pristine $t exits non-zero." >&2; PRISTINE_OK=0; }
    nf=$(failed_count "$LAST_OUT")
    if [ -z "$nf" ] || [ "$nf" -ne 0 ]; then
        echo "HARNESS DEFECT: pristine $t is not green (${nf:-no summary} failed)." >&2
        PRISTINE_OK=0
    else
        echo "pristine: $t -- $(passed_count "$LAST_OUT") passed, 0 failed"
    fi
done
[ "$PRISTINE_OK" -eq 1 ] || exit 99
echo "pristine: all three suites green -- mutation may begin"

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

# THE SEPARATOR. 0x1f (the "unit separator") cannot occur in either side of a
# mutation spec, so it is safe as a delimiter -- unlike '/', which C++ is FULL
# of and which killed a previous harness's perl s/// script.
#
# **IT MUST BE A REAL BYTE.** `$'\x1f'` produces it; `"\\x1f"` inside double
# quotes produces the four literal characters backslash-x-1-f, and Python's
# `split("\x1f")` then finds no separator and raises ValueError -- which is
# what the FIRST DRAFT of this harness did, and every mutation was reported
# HARNESS-DEFECT while the verdict table still looked plausible.
US=$'\x1f'

CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0; INERT=0
declare -a TABLE

# The verdict table is printed AFTER the inverted-verdict block, because that
# block can REWRITE M8's row (SURVIVED -> MEASURED-INERT). Printing it before
# would show a verdict the harness has already superseded.
print_table() {
    printf '%-6s %-46s %-42s %s\n' "ID" "MUTATION" "VERDICT" "DETAIL"
    printf '%-6s %-46s %-42s %s\n' "----" "----------------------------------------------" "------------------------------------------" "------"
    local row a b c d
    for row in "${TABLE[@]}"; do
        IFS='|' read -r a b c d <<< "$row"
        printf '%-6s %-46s %-42s %s\n' "$a" "$b" "$c" "${d:-}"
    done
}

# -----------------------------------------------------------------------------
# run_mutation <id> <description> <old> <new>
#
# A mutation is CAUGHT if ANY of the three suites fails. Which suite caught it
# is reported, because the three suites cover DIFFERENT decisions and a
# mutation caught only by the broadest one is weaker evidence than one caught
# by the suite that owns the decision.
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
        TABLE+=("$id|$desc|HARNESS-DEFECT (edit did not land)"); NOOP=$((NOOP+1)); return
    fi

    build_suites
    local rc=$?
    if [ "$rc" -eq 90 ]; then
        restore
        TABLE+=("$id|$desc|MUT-BROKEN (build failed -- NOT a verdict)"); BROKEN=$((BROKEN+1)); return
    fi

    local t caught_by="" suite_out nf
    for t in test_linear_algebra test_jacobi_eig test_galore_projection; do
        run_suite "$t"; rc=$?
        suite_out="$LAST_OUT"
        nf=$(failed_count "$suite_out")
        if [ "$rc" -ge 128 ]; then
            restore
            TABLE+=("$id|$desc|MUT-CRASH in $t (rc=$rc -- NOT a verdict)")
            CRASH=$((CRASH+1)); return
        fi
        if [ -n "$nf" ] && [ "$nf" -gt 0 ]; then
            caught_by="$t"
            break
        fi
    done

    if [ -n "$caught_by" ]; then
        local line
        line=$(echo "$suite_out" | grep -m1 "FAIL" | sed 's/^ *//' | cut -c1-70)
        TABLE+=("$id|$desc|CAUGHT by $caught_by|$line")
        CAUGHT=$((CAUGHT+1))
    else
        TABLE+=("$id|$desc|SURVIVED (all three suites stayed green)|")
        SURVIVED=$((SURVIVED+1))
    fi
    restore
}

echo
echo "=== linear_algebra mutations ==="

# --- M1: the rotation angle's sign (the 2026-10-28 defect, now in ONE place) --
run_mutation M1 "rotation angle: atan2(2*apq, aqq - app)" \
    "theta = 0.5 * std::atan2(2.0 * apq, app - aqq);" \
    "theta = 0.5 * std::atan2(2.0 * apq, aqq - app);"

# --- M2: drop the DESCENDING sort (the 2026-10-29 defect) --------------------
run_mutation M2 "drop the descending eigenpair sort" \
    "[&lam](size_t a, size_t b) { return lam[a] > lam[b]; });" \
    "[&lam](size_t a, size_t b) { return false; });"

# --- M3: drop the max(lam, 0) clamp (a NaN producer) -------------------------
run_mutation M3 "drop the max(lam,0) clamp in the matrix sqrt" \
    "const double root = std::sqrt(std::max(eig.eigenvalues(0, j), 0.0));" \
    "const double root = std::sqrt(eig.eigenvalues(0, j));"

# --- M4: the Frechet PRODUCT ORDER ------------------------------------------
run_mutation M4 "Frechet: S1.S2 instead of S1^(1/2).S2.S1^(1/2)" \
    "Tensor inner = S1h * S2 * S1h;" \
    "Tensor inner = S1 * S2;"

# --- M5: sort the eigenvalues but NOT the eigenvectors (the pairing bug) ------
run_mutation M5 "sort the eigenvalues but not the eigenvectors" \
    "out.eigenvectors(i, j) = Q(i, order[j]);" \
    "out.eigenvectors(i, j) = Q(i, j);"

# --- M6: drop the input SYMMETRISATION ---------------------------------------
run_mutation M6 "drop the input symmetrisation" \
    "const double v = 0.5 * (A(i, j) + A(j, i));" \
    "const double v = A(i, j);"

# --- M7: the mean-dimension guard --------------------------------------------
run_mutation M7 "drop the mean-dimension guard" \
    "if (mu1.cols != mu2.cols)" \
    "if (false)"

# --- M9: drop the FIRST covariance's dimension guard ------------------------
# (Added after the first harness run: the suite's needle was the bare word
# "dimension", which the SECOND covariance's guard message ALSO contains, so
# removing the mean-dimension guard went unnoticed. The suite now pins each
# dimension guard with its OWN message.)
run_mutation M9 "drop the first covariance's dimension guard" \
    "if (S1.rows != mu1.cols)" \
    "if (false)"

# --- M10: drop the SECOND covariance's dimension guard -----------------------
run_mutation M10 "drop the second covariance's dimension guard" \
    "if (S2.rows != mu2.cols)" \
    "if (false)"

# --- M8: the `app == aqq` degenerate branch -- MEASURED INERT ----------------
run_mutation M8 "drop the app==aqq degenerate branch" \
    "                if (std::abs(app - aqq) < 1e-30) {
                    // The degenerate branch: the atan2 would be 0/0.
                    theta = (apq >= 0.0) ? M_PI / 4.0 : -M_PI / 4.0;
                } else {
                    theta = 0.5 * std::atan2(2.0 * apq, app - aqq);
                }" \
    "                theta = 0.5 * std::atan2(2.0 * apq, app - aqq);"

echo
echo "=== caught $CAUGHT / survived $SURVIVED / broken $BROKEN / crashed $CRASH / noop $NOOP ==="

# -----------------------------------------------------------------------------
# THE INVERTED VERDICT FOR THE MEASURED-INERT MUTATION (M8).
#
# M8 SURVIVES the suites, and that is a CORRECT verdict rather than a coverage
# gap: with the corrected general formula, `atan2(2*apq, 0)` is EXACTLY
# +-pi/2, so the general branch already produces +-pi/4 -- bit-identical to
# what the special case hardcodes.
#
# **A SURVIVOR MUST NEVER BE SILENTLY DROPPED**, and an "inert" verdict must
# itself be checkable -- otherwise a future refactor that makes the branch
# load-bearing again would leave the harness reporting a stale "inert" that
# is no longer true. `tests/test_jacobi_inertness.cpp` re-derives BOTH angle
# variants INDEPENDENTLY of the library and FAILS if the inertness ever stops
# holding. This function INVERTS the verdict: it PASSES when the mutation is
# still inert and FAILS when it starts to matter.
# -----------------------------------------------------------------------------
echo ""
echo "=== INVERTED VERDICT: is M8 still provably inert? ==="
#
# **THE INVERTED VERDICT MAY ONLY DOWNGRADE A VERDICT THAT WAS ACTUALLY
# "SURVIVED".** The first draft of this harness ran the check
# unconditionally, so when M8 came back HARNESS-DEFECT (its mutation had
# never applied) the check still passed and the harness REPORTED
# "MEASURED-INERT" for a mutation that had never run -- and then decremented
# a SURVIVED counter that was already 0, printing "-1 survived". **A
# verdict about a mutation that did not run is not a verdict at all.**
M8_VERDICT=""
for row in "${TABLE[@]}"; do
    case "$row" in
        "M8|"*) M8_VERDICT="$row" ;;
    esac
done
case "$M8_VERDICT" in
    *"SURVIVED"*) ;;
    *)
        echo "M8  NOT-APPLICABLE  (M8's verdict was not SURVIVED -- the inertness claim cannot be scored)"
        echo "      M8's actual verdict: ${M8_VERDICT:-<missing>}"
        NOOP=$((NOOP+1))
        print_table
        echo ""
        echo "FINAL: $CAUGHT caught / $SURVIVED survived / $INERT measured-inert / $BROKEN broken / $CRASH crashed / $NOOP noop"
        exit 1
        ;;
esac
g++ -std=c++17 -O2 -Wall -Wextra -Iinclude -o "$SCRATCH/m8_inert" tests/test_jacobi_inertness.cpp 2>"$SCRATCH/build.log"
if [ $? -ne 0 ]; then
    echo "M8-INERT  HARNESS-BROKEN  (the inverted-verdict check did not compile)"
    BROKEN=$((BROKEN+1))
elif "$SCRATCH/m8_inert" > "$SCRATCH/m8_out.txt" 2>&1; then
    echo "M8  MEASURED-INERT  (the inverted-verdict check PASSES: the claim still holds)"
    grep -E "matrices tested|BIT-IDENTICAL|worst abs" "$SCRATCH/m8_out.txt" | sed 's/^/      /'
    # The survivor is EXPLAINED: downgrade it from "survived" to "inert".
    SURVIVED=$((SURVIVED-1)); INERT=$((INERT+1))
    declare -a NEW_TABLE=()
    local_new=""
    for row in "${TABLE[@]}"; do
        case "$row" in
            "M8|"*) local_new="M8|drop the app==aqq degenerate branch|MEASURED-INERT (the inverted verdict confirms it)|" ;;
            *) local_new="$row" ;;
        esac
        NEW_TABLE+=("$local_new")
    done
    TABLE=("${NEW_TABLE[@]}")
else
    echo "M8  NOT-INERT  (the inverted-verdict check FAILED -- the 'inert' verdict is now WRONG)"
    grep -E "\[FAIL\]|differing|worst abs" "$SCRATCH/m8_out.txt" | sed 's/^/      /'
fi

echo ""
print_table
echo ""
echo "FINAL: $CAUGHT caught / $SURVIVED survived / $INERT measured-inert / $BROKEN broken / $CRASH crashed / $NOOP noop"

[ "$SURVIVED" -eq 0 ] || { echo "NOTE: $SURVIVED unexplained survivor(s) -- these are REAL coverage gaps." >&2; exit 3; }
[ "$BROKEN" -eq 0 ] || { echo "NOTE: $BROKEN mutation(s) reported BROKEN/CRASH -- re-run in isolation." >&2; exit 1; }
[ "$NOOP" -eq 0 ] || { echo "NOTE: $NOOP harness defect(s)." >&2; exit 1; }
exit 0
