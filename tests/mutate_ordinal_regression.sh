#!/usr/bin/env bash
# =============================================================================
# tests/mutate_ordinal_regression.sh -- mutation harness for
# include/nn/layers/dense/ordinal_regression.{h,cpp}.
#
# PURPOSE: prove that tests/test_ordinal_regression.cpp actually CATCHES the
# defects it was written for. A green suite proves nothing on its own; the
# mutation has to fail.
#
# =============================================================================
# THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
# =============================================================================
#
#   M1  the interior class probability is the NAIVE difference of two
#       cumulatives instead of the stable product
#       -> the stable-form decision (FINDING 1). The naive form is EXACTLY 0
#          on the F3/F5 fixtures, where the truth is 2.685e-18.
#   M2  the stable product DROPS the e^{-v} factor
#       -> the stable-form identity.
#   M3  the (1 - c) factor is computed as (1.0 - c) instead of sigmoid(-a)
#       -> the stable evaluation of the 1-c factor (FINDING 2).
#   M4  the q term is dropped from the interior class's gap coefficient
#       -> the gap-dependence of an interior class's probability.
#   M5  the gap is recovered by DIFFERENCING the cut points instead of from
#       the raw parameter
#       -> the gap-absorbed decision. Differencing reports a gap of 0 on
#          the F4 fixture, where the truth is 3.7e-44.
#   M6  the ordering is NOT structural (the raw parameters ARE the cut points)
#       -> the structural-ordering decision.
#   M7  the gap shift is credited to ONE class only
#       -> the three-case d/dgap rule.
#   M8  the dL/deta of an interior class has the wrong sign on the c term
#       -> the direction of the eta gradient.
#   M9  the monotone direction is inverted
#       -> the monotonicity decision.
#   M10 the log(1 - e^{-gap}) term uses (1.0 - exp(-gap)) instead of expm1
#       -> the stable evaluation of the gap's log term. The naive form is
#          EXACTLY 0 for a tiny gap, so the interior class's log probability
#          becomes -inf.
#   M11 the input-width guard is removed
#       -> a mismatched input reaches the dot product instead of being
#          rejected.
#   M12 the gradient is computed in PROBABILITY space instead of log space
#       -> the log-space gradient decision (FINDING 2). The probability-space
#          derivative is EXACTLY 0 where the truth is about -1.
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
SRC="${REPO_ROOT}/include/nn/layers/dense/ordinal_regression.cpp"
BUILD_DIR="${REPO_ROOT}/build"
MUT_BUILD="${REPO_ROOT}/build/mutation/ordinal_regression"
TEST_SRC="${REPO_ROOT}/tests/test_ordinal_regression.cpp"

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
M1|the interior class probability is the NAIVE difference of two cumulatives|            out[i][k] = log_sigmoid(a[k]) + log_sigmoid(-a[k - 1]) +\n                        log1mexp_neg(gap[k - 1]);|            out[i][k] = log_sigmoid(a[k]) - log_sigmoid(a[k - 1]);
M2|the stable product DROPS the e^{-v} factor|            out[i][k] = log_sigmoid(a[k]) + log_sigmoid(-a[k - 1]) +\n                        log1mexp_neg(gap[k - 1]);|            out[i][k] = log_sigmoid(a[k]) + log_sigmoid(a[k - 1]) +\n                        log1mexp_neg(gap[k - 1]);
M3|the (1 - c) factor is computed as (1.0 - c) instead of sigmoid(-a)|            e[j] = sigmoid(-a[j]);|            e[j] = 1.0 - c[j];
M4|the q term is dropped from the interior class's gap coefficient|                special[k] += g * (e[k] + q[k - 1]);|                special[k] += g * (e[k]);
M5|the gap is recovered by DIFFERENCING the cut points|        gap[j] = softplus(raw_cuts_[0][j + 1]);\n        q[j] = gap_q(gap[j]);\n        theta[j + 1] = theta[j] + gap[j];|        gap[j] = theta[j + 1] - theta[j];\n        q[j] = gap_q(gap[j]);\n        theta[j + 1] = theta[j] + gap[j];
M6|the ordering is NOT structural: the raw parameters ARE the cut points|        gap[j] = softplus(raw_cuts_[0][j + 1]);\n        theta[j + 1] = theta[j] + gap[j];|        gap[j] = raw_cuts_[0][j + 1] - raw_cuts_[0][j];\n        theta[j + 1] = theta[j] + gap[j];
M7|the gap shift is credited to ONE class only|        suffix += bulk[j + 2];\n        dL_dgap[j] = suffix + special[j + 1];|        suffix += bulk[j + 2];\n        dL_dgap[j] = special[j + 1];
M8|the dL/deta of an interior class has the wrong sign on the c term|                acc_eta += g * (c[k - 1] - e[k]);|                acc_eta += g * (e[k] - c[k - 1]);
M9|the monotone direction is inverted (a higher eta moves mass to LOWER classes)|    double eta = 0.0;\n        for (size_t d = 0; d < D; ++d) eta += input[i][d] * beta_[0][d];\n\n        // ---- the a values ----|    double eta = 0.0;\n        for (size_t d = 0; d < D; ++d) eta -= input[i][d] * beta_[0][d];\n\n        // ---- the a values ----
M10|the log(1 - e^{-gap}) term uses (1.0 - exp(-gap)) instead of expm1|    return std::log(-std::expm1(-gap));|    return std::log(1.0 - std::exp(-gap));
M11|the input-width guard is removed|    if (input.cols != in_features_) {\n        throw std::invalid_argument(\n            "OrdinalRegression: input width " +\n            std::to_string(input.cols) + " does not match in_features " +\n            std::to_string(in_features_));\n    }|    if (false) {\n        throw std::invalid_argument(\n            "OrdinalRegression: input width " +\n            std::to_string(input.cols) + " does not match in_features " +\n            std::to_string(in_features_));\n    }
M12|the gradient is computed in PROBABILITY space instead of log space|                acc_eta += g * (-e[0]);|                acc_eta += g * (-c[0] * (1.0 - c[0]));
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
MUT_OBJ="${MUT_BUILD}/ordinal_regression.o"
LINK_OBJS=""
for o in ${LIB_OBJS}; do
    case "$o" in
        */layers/dense/ordinal_regression.o) ;;
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
    rm -f "${MUT_OBJ}" "${MUT_BUILD}/test_ordinal_regression"
    if ! g++ -std=c++17 -O2 -Wall -Wextra -march=native -I"${REPO_ROOT}/include" \
            -c "${SRC}" -o "${MUT_OBJ}" 2> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the mutated TU did not compile -- a BUILD verdict, not a coverage one"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ -std=c++17 -O2 -I"${REPO_ROOT}/include" \
            -c "${TEST_SRC}" -o "${MUT_BUILD}/test_ordinal_regression.o" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the test did not compile -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ "${MUT_BUILD}/test_ordinal_regression.o" "${MUT_OBJ}" ${LINK_OBJS} \
            -o "${MUT_BUILD}/test_ordinal_regression" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the link failed -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi

    # --- run the suite.
    "${MUT_BUILD}/test_ordinal_regression" > "${MUT_BUILD}/run.log" 2>&1
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
