#!/usr/bin/env bash
# =============================================================================
# tests/mutate_quantile_regression.sh -- mutation harness for
# include/nn/layers/dense/quantile_regression.{h,cpp}.
#
# PURPOSE: prove that tests/test_quantile_regression.cpp actually CATCHES the
# defects it was written for. A green suite proves nothing on its own; the
# mutation has to fail.
#
# =============================================================================
# THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
# =============================================================================
#
#   M1  the pinball loss uses the OVER-prediction branch at e >= 0 (the tie
#       convention is inverted)
#       -> the tie convention, which must MATCH the shipped QuantileLoss.
#          This is the mutation that breaks the bit-for-bit K = 1 equality.
#   M2  the gradient's over-prediction branch has the wrong sign
#       -> the direction of the gradient.
#   M3  the loss is reduced by dividing by (N * K) instead of by N
#       -> the reduction ("mean over the batch, sum over the levels"). The
#          shipped QuantileLoss divides by N, and the K = 1 equality pins it.
#   M4  the gradient is NOT divided by N
#       -> the same reduction, on the gradient side.
#   M5  the level index is off by one: the gradient uses quantiles_[k - 1]
#       -> the per-level association between a column and its level.
#   M6  the gradient mixes the LEVELS: every column uses quantiles_[0]
#       -> the LEVEL DECOUPLING, which is what makes the classical
#          independent fit possible.
#   M7  the bias gradient is not accumulated (it is ASSIGNED)
#       -> the accumulation contract that every layer in the repo follows.
#   M8  the weight gradient is not accumulated (it is ASSIGNED)
#       -> the same contract, on the other parameter.
#   M9  the input gradient omits the weight factor
#       -> the chain rule for dL/d(input).
#  M10  the crossing diagnostic counts NON-adjacent pairs instead of adjacent
#       -> the adjacency rule. A non-adjacent count is zero exactly when the
#          row is monotone too, so the DIFFERENCE is in the MAGNITUDE -- and
#          the magnitude is what the fixture pins.
#  M11  the monotone projection sorts DESCENDING
#       -> the direction of the rearrangement.
#  M12  the monotone projection sorts the COLUMNS instead of the ROWS
#       -> the per-row (not per-column) rule.
#  M13  the input-width guard is removed
#       -> a mismatched input reaches the product instead of being rejected.
#  M14  the (N, 1) shape guard on y is removed
#       -> the shape rule that distinguishes this layer from the shipped
#          elementwise QuantileLoss.
#  M15  the levels are not validated as strictly increasing
#       -> the constructor's ordering guard.
#  M16  the levels start with DIFFERENT slopes, so the model begins life
#       crossing
#       -> the initialisation decision. The levels are ORDERED by level, so
#          a per-level initialisation would give them mutually inconsistent
#          slopes: the model would start life CROSSING, and the crossing
#          diagnostic would be measuring the INITIALISATION rather than the
#          fit. **A UNIFORM non-zero fill does NOT test this** -- every level
#          would still have the SAME slope, so the model would still start
#          flat. The mutation has to make the levels DIFFER, and the first
#          draft of this harness used a uniform fill and was reported
#          SURVIVED, correctly: the fixture could not see it.
#          The suite's check was also VACUOUS for the same reason -- it used
#          an all-zero INPUT, so the predictions were zero for ANY weights.
#          Both the mutation and the fixture were fixed.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
# =============================================================================
#
#   1. `restore()` must NEVER delete its backup. It is called once PER
#      MUTATION and once on EXIT, so a self-cleaning restore works exactly
#      once and every later restore is a silent no-op -- leaving a MUTANT
#      IN THE TREE. Cleanup happens exactly once, on EXIT, and restore() is
#      ATOMIC (`cp tmp && mv`), because a plain `cp` truncates the
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
SRC="${REPO_ROOT}/include/nn/layers/dense/quantile_regression.cpp"
BUILD_DIR="${REPO_ROOT}/build"
MUT_BUILD="${REPO_ROOT}/build/mutation/quantile_regression"
TEST_SRC="${REPO_ROOT}/tests/test_quantile_regression.cpp"

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
M1|the pinball loss uses the OVER-prediction branch at e >= 0 (the tie convention is inverted)|inline double pinball(double e, double tau) {\n    if (e >= 0.0) return e * tau;\n    return e * (tau - 1.0);\n}|inline double pinball(double e, double tau) {\n    if (e >= 0.0) return e * (tau - 1.0);\n    return e * tau;\n}
M2|the gradient's over-prediction branch has the wrong sign|inline double pinball_grad_pred(double e, double tau) {\n    if (e >= 0.0) return -tau;\n    return 1.0 - tau;\n}|inline double pinball_grad_pred(double e, double tau) {\n    if (e >= 0.0) return -tau;\n    return tau - 1.0;\n}
M3|the loss is reduced by dividing by (N * K) instead of by N|    out[0][0] = total / static_cast<double>(N);|    out[0][0] = total / static_cast<double>(N * K);
M4|the gradient is NOT divided by N|            grad[i][k] = pinball_grad_pred(yi - pred[i][k], quantiles_[k]) * inv_N;|            grad[i][k] = pinball_grad_pred(yi - pred[i][k], quantiles_[k]);
M5|the level index is off by one: the gradient uses quantiles_[k - 1]|            grad[i][k] = pinball_grad_pred(yi - pred[i][k], quantiles_[k]) * inv_N;|            grad[i][k] = pinball_grad_pred(yi - pred[i][k], quantiles_[k == 0 ? 0 : k - 1]) * inv_N;
M6|the gradient mixes the LEVELS: every column uses quantiles_[0]|            grad[i][k] = pinball_grad_pred(yi - pred[i][k], quantiles_[k]) * inv_N;|            grad[i][k] = pinball_grad_pred(yi - pred[i][k], quantiles_[0]) * inv_N;
M7|the bias gradient is not accumulated (it is ASSIGNED)|        grad_bias_[0][k] += sum;|        grad_bias_[0][k] = sum;
M8|the weight gradient is not accumulated (it is ASSIGNED)|            grad_weights_[r][c] += grad_w[r][c];|            grad_weights_[r][c] = grad_w[r][c];
M9|the input gradient omits the weight factor|    Tensor grad_input = grad_output * weights_;|    Tensor grad_input = grad_output * Tensor(grad_output.cols, 1);
M10|the crossing diagnostic counts ALL pairs, not just adjacent ones|        for (size_t k = 0; k + 1 < K; ++k) {\n            if (pred[i][k] > pred[i][k + 1]) ++count;\n        }|        for (size_t k = 0; k + 1 < K; ++k) {\n            for (size_t l = k + 1; l < K; ++l) {\n                if (pred[i][k] > pred[i][l]) ++count;\n            }\n        }|
M11|the monotone projection sorts DESCENDING|        std::sort(row.begin(), row.end());|        std::sort(row.begin(), row.end(), std::greater<double>());
M12|the monotone projection sorts the COLUMNS instead of the ROWS|    Tensor out = pred.clone();\n    std::vector<double> row(K);\n    for (size_t i = 0; i < N; ++i) {\n        for (size_t k = 0; k < K; ++k) row[k] = pred[i][k];\n        std::sort(row.begin(), row.end());\n        for (size_t k = 0; k < K; ++k) out[i][k] = row[k];\n    }\n    return out;|    Tensor out = pred.clone();\n    std::vector<double> col(N);\n    for (size_t k = 0; k < K; ++k) {\n        for (size_t i = 0; i < N; ++i) col[i] = pred[i][k];\n        std::sort(col.begin(), col.end());\n        for (size_t i = 0; i < N; ++i) out[i][k] = col[i];\n    }\n    return out;
M13|the input-width guard is removed|    if (input.cols != in_features_) {\n        throw std::invalid_argument(\n            "QuantileRegression: input width " + std::to_string(input.cols) +\n            " does not match in_features " + std::to_string(in_features_));\n    }|    if (false) {\n        throw std::invalid_argument(\n            "QuantileRegression: input width " + std::to_string(input.cols) +\n            " does not match in_features " + std::to_string(in_features_));\n    }
M14|the (N, 1) shape guard on y is removed|    if (y.cols != 1) {\n        throw std::invalid_argument(\n            "QuantileRegression::pinball_loss: y must be (N, 1) -- ONE "\n            "response per row, compared against all " + std::to_string(K) +\n            " levels -- but it has " + std::to_string(y.cols) + " columns");\n    }|    if (false) {\n        throw std::invalid_argument(\n            "QuantileRegression::pinball_loss: y must be (N, 1) -- ONE "\n            "response per row, compared against all " + std::to_string(K) +\n            " levels -- but it has " + std::to_string(y.cols) + " columns");\n    }
M15|the levels are not validated as strictly increasing|        if (k > 0 && !(t > quantiles_[k - 1])) {|        if (false) {
M16|the levels start with DIFFERENT slopes, so the model begins life crossing|    weights_.fill(0.0);\n    bias_.fill(0.0);|    for (size_t k = 0; k < weights_.rows; ++k) {\n        for (size_t d = 0; d < weights_.cols; ++d) {\n            weights_[k][d] = 0.1 * static_cast<double>(k + 1);\n        }\n    }\n    bias_.fill(0.0);|
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
MUT_OBJ="${MUT_BUILD}/quantile_regression.o"
LINK_OBJS=""
for o in ${LIB_OBJS}; do
    case "$o" in
        */layers/dense/quantile_regression.o) ;;
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
    rm -f "${MUT_OBJ}" "${MUT_BUILD}/test_quantile_regression"
    if ! g++ -std=c++17 -O2 -Wall -Wextra -march=native -I"${REPO_ROOT}/include" \
            -c "${SRC}" -o "${MUT_OBJ}" 2> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the mutated TU did not compile -- a BUILD verdict, not a coverage one"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ -std=c++17 -O2 -I"${REPO_ROOT}/include" \
            -c "${TEST_SRC}" -o "${MUT_BUILD}/test_quantile_regression.o" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the test did not compile -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ "${MUT_BUILD}/test_quantile_regression.o" "${MUT_OBJ}" ${LINK_OBJS} \
            -o "${MUT_BUILD}/test_quantile_regression" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the link failed -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi

    # --- run the suite.
    "${MUT_BUILD}/test_quantile_regression" > "${MUT_BUILD}/run.log" 2>&1
    rc=$?
    n_fail=$(grep -c '\[FAIL\]' "${MUT_BUILD}/run.log" 2>/dev/null || echo 0)
    first=$(grep -m1 '\[FAIL\]' "${MUT_BUILD}/run.log" | cut -c1-140)

    if [ $rc -eq 0 ]; then
        echo "${mid} SURVIVED  ${desc}"
        SURVIVED=$((SURVIVED + 1))
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
