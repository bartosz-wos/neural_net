#!/usr/bin/env bash
# =============================================================================
# tests/mutate_isotonic.sh -- mutation harness for
# include/nn/utils/isotonic_regression.{h,cpp}.
#
# PURPOSE: prove that tests/test_isotonic.cpp and
# tests/test_isotonic_differential.cpp actually CATCH the defects they were
# written for. A green suite proves nothing on its own; the mutation has to
# fail.
#
# =============================================================================
# THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
# =============================================================================
#
#   M1  the tie-collapse uses the UNWEIGHTED mean of y
#       -> the tie-collapse rule. Section 1.
#   M2  the tie-collapse does not SUM the weights
#       -> the tie-collapse rule, from the other side. Section 1.
#   M3  the violation test compares the LAST block against the FIRST (a
#       global-minimum shortcut) rather than against its ADJACENT
#       predecessor
#       -> the adjacency rule. Section 2.
#   M4  the merged block's value is the mean of the two block MEANS, not of
#       the block's points
#       -> the pooling rule. Section 2. THE MOST IMPORTANT ONE: it is
#          invisible on a fixture whose blocks all have the same size.
#   M5  the prediction is the STEP function, not the linear interpolation
#       -> the step-vs-interpolation distinction. Section 3.
#   M6  the trimming is not applied (all knots retained)
#       -> the trimming rule. Section 4.
#   M7  zero-weight points are kept instead of dropped
#       -> the zero-weight rule. Section 1b.
#   M8  the decreasing direction is "negate and fit increasing"
#       -> the decreasing-direction rule. Section 2.
#   M9  out_of_bounds=Clip returns the MEAN of the end values instead of
#       the end values
#       -> the out-of-bounds rule. Section 5.
#   M10 the x/y length-mismatch guard is REMOVED
#       -> a mismatched input reaches the read loop instead of being
#          rejected. Section 7. **MEASURED, NOT ASSUMED**: the verdict this
#          mutation produces is whatever the run reports -- the guard
#          removal is exactly the kind of edit whose outcome is not
#          predictable from reading the code, because the mismatch then
#          reaches an index that the removed guard used to protect.
#   M11 the non-finite guard is removed
#       -> a NaN propagates into the fit instead of being rejected.
#          Section 7.
#   M12 the report's "after" is computed from the BEFORE probabilities
#       -> the report measures nothing. Section 6.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
# =============================================================================
#
#   1. `restore()` must NEVER delete its backup. It is called once PER
#      MUTATION and once on EXIT, so a self-cleaning restore works exactly
#      once and every later restore is a silent no-op -- leaving a MUTANT
#      IN THE TREE. Cleanup happens exactly once, on EXIT, and restore() is
#      ATOMIC (`cp tmp && mv -f`), because a plain `cp` truncates the
#      destination first, so a concurrent reader can splice two generations
#      together.
#   2. The link set comes from `make print-lib-objs`, NEVER from
#      regex-parsing the Makefile. `LIB_OBJS` is a RECURSIVE make variable
#      whose literal text contains no `.o` filenames, so a regex yields an
#      empty set and every mutation would score a false MUT-BROKEN -- one
#      defect reported as N verdicts. The count is asserted >= 10.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET. It legitimately IS
#      in the library's list (it is the TU being mutated), so the mutated TU
#      is compiled SEPARATELY and swapped in for the library's copy.
#   4. `MUT-BROKEN` means "the build failed". It is a SEPARATE CLASS and is
#      never counted as a pass. A mutation reported BROKEN is re-run in
#      isolation before it is believed.
#   5. `rc >= 128` is a CRASH, which is a THIRD verdict class: the binary
#      compiled and then died, which says NOTHING about whether the suite
#      covers the behaviour. It is not reported as CAUGHT.
#   6. `rm -f` the intermediates before every build. A stale `.o` from the
#      previous mutation produces a phantom failure.
#   7. NEVER run a build while this harness is running: two harnesses that
#      mutate the same source file race on `build/*.o`.
#   8. The ANCHOR must be UNIQUE. A line that appears twice makes the edit
#      a coin flip, and a coin flip is not a verdict. Each anchor's count
#      is asserted to be exactly 1 before the edit.
#   9. The edits are done with a PYTHON script, not perl/sed: this repo has
#      been bitten by perl's s/// delimiter (the anchors are full of "/")
#      and by sed interpreting a newline inside a pattern. Python's
#      str.count / str.replace has no delimiter concept at all.
#
# =============================================================================
# THE VERDICT CLASSES
# =============================================================================
#
#   CAUGHT     the suite FAILED. This is the only good outcome.
#   SURVIVED   the suite PASSED. This is a hole in the SUITE, and it is
#              reported as such -- never as a pass.
#   MUT-BROKEN the BUILD failed. Not a verdict about the suite.
#   MUT-CRASH  the binary died (rc >= 128). Not a verdict about the suite.
#   MUT-NOOP   the mutation did not apply. A harness defect.
#   MUT-ANCHOR the anchor was not unique. A harness defect.
#
# =============================================================================

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/include/nn/utils/isotonic_regression.cpp"
BUILD_DIR="${REPO_ROOT}/build"
MUT_BUILD="${REPO_ROOT}/build/mutation/isotonic"
TEST_SRC="${REPO_ROOT}/tests/test_isotonic.cpp"
DIFF_SRC="${REPO_ROOT}/tests/test_isotonic_differential.cpp"

CAUGHT=0
SURVIVED=0
BROKEN=0
CRASHED=0
NOOP=0
ANCHOR_BAD=0

# --- the backup. restore() NEVER deletes it; cleanup does, on EXIT. -------
BACKUP="$(mktemp)"
cp "${SRC}" "${BACKUP}"

restore() {
    # ATOMIC: cp to a temp then mv, so a concurrent reader never sees a
    # half-written source.
    cp "${BACKUP}" "${SRC}.tmp" && mv -f "${SRC}.tmp" "${SRC}"
}

cleanup() {
    restore
    rm -f "${BACKUP}"
    rm -rf "${MUT_BUILD}"
}
trap cleanup EXIT INT TERM

# =============================================================================
# The mutation table.
#   id | description | python search | python replacement
# =============================================================================
read -r -d '' MUTATIONS <<'PYEOF'
M1|the tie-collapse uses the UNWEIGHTED mean of y|            sum_wy += ws[order[j]] * ys[order[j]];|            sum_wy += ys[order[j]];
M2|the tie-collapse does not SUM the weights|            sum_w += ws[order[j]];|            sum_w += 1.0;
M3|the violation test compares the LAST block against the FIRST (a global-minimum shortcut)|            const Block& prev = stack[stack.size() - 2];|            const Block& prev = stack[0];
M4|the merged block's value is the mean of the two block MEANS|            a.sum_wy += stack.back().sum_wy;\n            a.sum_w += stack.back().sum_w;|            a.sum_wy = 0.5 * (block_value(a) + block_value(stack.back())) * (a.sum_w + stack.back().sum_w);\n            a.sum_w = a.sum_w + stack.back().sum_w;
M5|the prediction is the STEP function, not the linear interpolation|            const double t = (q - x0) / (x1 - x0);\n            out(i, 0) = y0 + t * (y1 - y0);|            out(i, 0) = y0;
M6|the trimming is not applied (all knots retained)|        const bool redundant =\n            interior && fitted[g] == fitted[g - 1] && fitted[g] == fitted[g + 1];\n        if (redundant) continue;|        const bool redundant = false;\n        if (redundant) continue;
M7|zero-weight points are kept instead of dropped|        if (w <= 0.0) continue;   // zero-weight points are DROPPED|        if (w < 0.0) continue;
M8|the decreasing direction is 'negate and fit increasing'|            const bool violation = increasing ? (vp > vl) : (vp < vl);|            const bool violation = (vp > vl);
M9|out_of_bounds=Clip returns the MEAN of the end values|                    out(i, 0) = (q < x_min) ? fit.y_knots.front() : fit.y_knots.back();|                    out(i, 0) = 0.5 * (fit.y_knots.front() + fit.y_knots.back());
M10|the x/y length-mismatch guard is removed|    if (x.rows != y.rows) {\n        throw std::invalid_argument(std::string(fn) +\n                                    " requires x and y to have the same number of rows");\n    }|    if (false) {\n        throw std::invalid_argument(std::string(fn) +\n                                    " requires x and y to have the same number of rows");\n    }
M11|the non-finite guard is removed|    require_finite_column(x, "x", fn);\n    require_finite_column(y, "y", fn);|    require_finite_column(y, "y", fn);
M12|the report's 'after' is computed from the BEFORE probabilities|    rep.ece_after = expected_calibration_error(eval_labels, calibrated, n_bins,\n                                                1.0);|    rep.ece_after = expected_calibration_error(eval_labels, eval_scores, n_bins,\n                                                1.0);
PYEOF

# =============================================================================
# The link set, from `make print-lib-objs` -- NEVER from a regex over the
# Makefile. A regex yields ZERO objects (the variable is recursive) and
# every mutation would then score a false MUT-BROKEN.
# =============================================================================
LIB_OBJS="$(cd "${REPO_ROOT}" && make -s print-lib-objs | tr '\n' ' ')"
LIB_COUNT="$(echo "${LIB_OBJS}" | wc -w)"
if [ "${LIB_COUNT}" -lt 10 ]; then
    echo "MUT-HARNESS-BROKEN: the link set has only ${LIB_COUNT} objects (must be >= 10)"
    echo "A regex over the Makefile would have produced an EMPTY set here, and"
    echo "every mutation would score a false MUT-BROKEN."
    exit 2
fi
echo "link set: ${LIB_COUNT} objects"

# The mutated TU is dropped from the library's list and compiled separately.
MUT_OBJ="${MUT_BUILD}/isotonic_regression.o"
LINK_OBJS=""
for o in ${LIB_OBJS}; do
    case "$o" in
        */utils/isotonic_regression.o) ;;
        *) LINK_OBJS="${LINK_OBJS} $o" ;;
    esac
done
if [ "$(echo "${LINK_OBJS}" | wc -w)" -lt 10 ]; then
    echo "MUT-HARNESS-BROKEN: the link set minus the mutated object has too few members"
    exit 2
fi

mkdir -p "${MUT_BUILD}"

# =============================================================================
# The main loop.
# =============================================================================
while IFS='|' read -r mid desc search repl; do
    [ -z "${mid}" ] && continue

    # --- restore FIRST. Every iteration starts from the pristine source.
    restore

    # --- apply the mutation with python (no shell delimiter hazards).
    python3 - "${SRC}" "${search}" "${repl}" <<'PYEOF'
import sys
path, search, repl = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
search = search.replace("\\n", "\n")
repl = repl.replace("\\n", "\n")
count = src.count(search)
if count != 1:
    # NOT unique: the anchor is ambiguous, and an ambiguous edit is a coin
    # flip, not a verdict.
    sys.stderr.write("ANCHOR count=%d\n" % count)
    sys.exit(3)
open(path, "w").write(src.replace(search, repl))
PYEOF
    rc_edit=$?
    if [ $rc_edit -eq 3 ]; then
        echo "${mid} MUT-ANCHOR  the anchor is not unique -- harness defect, not a verdict"
        ANCHOR_BAD=$((ANCHOR_BAD + 1))
        continue
    elif [ $rc_edit -ne 0 ]; then
        echo "${mid} MUT-NOOP  the edit failed -- harness defect, not a verdict"
        NOOP=$((NOOP + 1))
        continue
    fi

    # --- checksum: a mutation that did not apply is a harness defect.
    if cmp -s "${SRC}" "${BACKUP}"; then
        echo "${mid} MUT-NOOP  the source is unchanged -- harness defect"
        NOOP=$((NOOP + 1))
        continue
    fi

    # --- build. The mutated TU is compiled separately and swapped in.
    rm -f "${MUT_OBJ}" "${MUT_BUILD}/test_isotonic" "${MUT_BUILD}/test_isotonic_differential"
    if ! g++ -std=c++17 -O2 -Wall -Wextra -march=native -I"${REPO_ROOT}/include" \
            -c "${SRC}" -o "${MUT_OBJ}" 2> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the mutated TU did not compile -- a BUILD verdict, not a coverage one"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ -std=c++17 -O2 -I"${REPO_ROOT}/include" \
            -c "${TEST_SRC}" -o "${MUT_BUILD}/test_isotonic.o" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the test did not compile -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ -std=c++17 -O2 -I"${REPO_ROOT}/include" \
            -c "${DIFF_SRC}" -o "${MUT_BUILD}/test_isotonic_differential.o" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the differential test did not compile -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ "${MUT_BUILD}/test_isotonic.o" "${MUT_OBJ}" ${LINK_OBJS} \
            -o "${MUT_BUILD}/test_isotonic" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the link failed -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ "${MUT_BUILD}/test_isotonic_differential.o" "${MUT_OBJ}" ${LINK_OBJS} \
            -o "${MUT_BUILD}/test_isotonic_differential" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the differential link failed -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi

    # --- run BOTH suites. A mutation is CAUGHT if EITHER suite fails:
    # --- the pinned suite and the randomised differential one.
    "${MUT_BUILD}/test_isotonic" > "${MUT_BUILD}/run.log" 2>&1
    rc=$?
    n_fail=$(grep -c '\[FAIL\]' "${MUT_BUILD}/run.log" 2>/dev/null || echo 0)
    first=$(grep -m1 '\[FAIL\]' "${MUT_BUILD}/run.log" | cut -c1-120)

    if [ $rc -eq 0 ]; then
        # The PINNED suite passed. The DIFFERENTIAL one is the second
        # chance, and it is the randomised sweep -- it sees the tie-heavy
        # weighted shapes the pinned fixtures do not.
        "${MUT_BUILD}/test_isotonic_differential" > "${MUT_BUILD}/run_diff.log" 2>&1
        rc_diff=$?
        n_fail_diff=$(grep -c '\[FAIL\]' "${MUT_BUILD}/run_diff.log" 2>/dev/null || echo 0)
        first_diff=$(grep -m1 '\[FAIL\]' "${MUT_BUILD}/run_diff.log" | cut -c1-120)
        if [ $rc_diff -eq 0 ]; then
            echo "${mid} SURVIVED  ${desc}"
            SURVIVED=$((SURVIVED + 1))
        elif [ $rc_diff -ge 128 ]; then
            echo "${mid} MUT-CRASH  ${desc}  (rc=${rc_diff})"
            CRASHED=$((CRASHED + 1))
        else
            echo "${mid} CAUGHT  ${desc}  [differential: ${n_fail_diff} failures]"
            echo "      ${first_diff}"
            CAUGHT=$((CAUGHT + 1))
        fi
    elif [ $rc -ge 128 ]; then
        # A CRASH is a THIRD verdict class: the binary compiled and then
        # died, which says nothing about whether the suite covers the
        # behaviour.
        echo "${mid} MUT-CRASH  ${desc}  (rc=${rc})"
        CRASHED=$((CRASHED + 1))
    else
        echo "${mid} CAUGHT  ${desc}  [${n_fail} failures]"
        echo "      ${first}"
        CAUGHT=$((CAUGHT + 1))
    fi
done <<< "${MUTATIONS}"

restore

echo ""
echo "=== SUMMARY: ${CAUGHT} caught, ${SURVIVED} survived, ${BROKEN} broken, ${CRASHED} crashed, ${NOOP} noop, ${ANCHOR_BAD} anchor ==="
if [ ${SURVIVED} -gt 0 ]; then
    echo "SURVIVORS ARE HOLES IN THE SUITE:"
    exit 1
fi
if [ ${BROKEN} -gt 0 ] || [ ${CRASHED} -gt 0 ] || [ ${NOOP} -gt 0 ] || [ ${ANCHOR_BAD} -gt 0 ]; then
    echo "HARNESS DEFECTS OR UNVERDICTED MUTATIONS (not passes):"
    exit 1
fi
exit 0
