#!/usr/bin/env bash
# =============================================================================
# tests/mutate_rank_correlation.sh -- mutation harness for
# utils/rank_correlation.{h,cpp}.
#
# PURPOSE: prove that tests/test_rank_correlation.cpp actually CATCHES the
# defects it was written for. A green suite proves nothing on its own; the
# mutation has to fail.
#
# =============================================================================
# THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
# =============================================================================
#
#   M1  the mid-ranks use the FIRST rank of a tied group -> the rank
#       convention. Section 1.
#   M2  tau-b counts the JOINT ties in NEITHER xtie NOR ytie -> the tie
#       correction. Section 3. THE MOST IMPORTANT ONE: it is invisible on
#       a fixture with no ties, so it needs a fixture with JOINT ties.
#   M3  the normal-scores plotting position is Blomqvist's -> the
#       plotting-position convention. Section 4.
#   M4  the population tau is (1/pi)asin(rho) -> a factor-2 error in a
#       closed form. Section 6.
#   M5  the population Spearman drops the /2 inside the arcsine -> a
#       dropped argument in a closed form. Section 6.
#   M6  the copula correlation is computed from the RAW values -> the
#       marginal-freeness property is lost, which is what makes it a
#       COPULA statistic. Section 5.
#   M7  Spearman uses the d^2 shortcut -> the shortcut is only valid
#       without ties. Section 2.
#   M8  the length-mismatch guard is removed -> a mismatched input is read
#       out of bounds instead of rejected. Section 7.
#   M9  the constant-sample case returns 0 instead of NaN -> a confident
#       wrong answer where the truth is "undefined". Section 7.
#   M10 the one-pass (uncen'tred) Pearson form is used -> the same
#       quantity in exact arithmetic, catastrophically different in
#       floating point on large-offset data. Section 2/8.
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
#      empty set and every mutation scores MUT-BROKEN -- one defect
#      reported as N verdicts. The count is asserted >= 10.
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
SRC="${REPO_ROOT}/include/nn/utils/rank_correlation.cpp"
BUILD_DIR="${REPO_ROOT}/build"
MUT_BUILD="${REPO_ROOT}/build/mutation/rank_correlation"
TEST_SRC="${REPO_ROOT}/tests/test_rank_correlation.cpp"

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
# The edits are done with a PYTHON script, not perl/sed: this repo has been
# bitten by perl's s/// delimiter (the anchors are full of "/") and by sed
# interpreting a newline inside a pattern. Python's str.count / str.replace
# has no delimiter concept at all.
# =============================================================================
read -r -d '' MUTATIONS <<'PYEOF'
M1|the mid-ranks use the FIRST rank of a tied group|        const double avg_rank = 0.5 * (static_cast<double>(i) + 1.0 +\n                                       static_cast<double>(j) + 1.0);|        const double avg_rank = static_cast<double>(i + 1);
M2|tau-b counts the JOINT ties in NEITHER xtie NOR ytie|                xtie += 1.0;\n                ytie += 1.0;\n                ntie += 1.0;|                ntie += 1.0;
M3|the normal-scores plotting position is Blomqvist's|const double p = (ranks[i] - 0.5) / n;|const double p = (ranks[i] - 0.375) / (n + 0.25);
M4|the population tau is (1/pi)asin(rho)|return 2.0 / M_PI * std::asin(rho);|return 1.0 / M_PI * std::asin(rho);
M5|the population Spearman drops the /2 inside the arcsine|return 6.0 / M_PI * std::asin(rho / 2.0);|return 6.0 / M_PI * std::asin(rho);
M6|the copula correlation is computed from the RAW values|return pearson_of(normal_scores_of(a), normal_scores_of(b));|return pearson_of(a, b);
M7|Spearman uses the d^2 shortcut|return pearson_of(mid_ranks_of(a), mid_ranks_of(b));|return 1.0 - 6.0 * d2_sum(mid_ranks_of(a), mid_ranks_of(b)) / (static_cast<double>(a.size()) * (static_cast<double>(a.size()) * a.size() - 1.0));
M8|the length-mismatch guard is removed|    require_same_length(a, b, "spearman_rho");\n    require_at_least_two(a, "spearman_rho");|    require_at_least_two(a, "spearman_rho");
M9|the constant-sample case returns 0 instead of NaN|        // A constant sample: the correlation is UNDEFINED (0/0). NaN is the\n        // honest answer and is what scipy.stats.spearmanr returns.\n        return std::numeric_limits<double>::quiet_NaN();|        // A constant sample: the correlation is UNDEFINED (0/0). NaN is the\n        // honest answer and is what scipy.stats.spearmanr returns.\n        return 0.0;
M10|the one-pass (uncentred) Pearson form is used|    double num = 0.0;\n    double saa = 0.0;\n    double sbb = 0.0;\n    for (size_t i = 0; i < n; ++i) {\n        const double da = a[i] - mean_a;\n        const double db = b[i] - mean_b;\n        num += da * db;\n        saa += da * da;\n        sbb += db * db;\n    }|    double num = 0.0;\n    double saa = 0.0;\n    double sbb = 0.0;\n    for (size_t i = 0; i < n; ++i) {\n        num += a[i] * b[i];\n        saa += a[i] * a[i];\n        sbb += b[i] * b[i];\n    }\n    num = num - static_cast<double>(n) * mean_a * mean_b;\n    saa = saa - static_cast<double>(n) * mean_a * mean_a;\n    sbb = sbb - static_cast<double>(n) * mean_b * mean_b;
PYEOF

# --- the helper the M7 mutation needs. It is ADDED to the source as part of
# --- that mutation, because a mutation that needs a new symbol is a
# --- TWO-PART edit and doing it as one is what makes the harness honest.
add_d2_helper() {
    python3 - "$SRC" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
helper = '''
// The sum of squared differences of two equal-length samples.
double d2_sum(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

'''
anchor = "// A correlation coefficient is in [-1, 1]."
if src.count(anchor) != 1:
    sys.stderr.write("d2_helper: anchor not unique\n")
    sys.exit(2)
src = src.replace(anchor, helper + anchor)
open(path, "w").write(src)
PYEOF
}

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
MUT_OBJ="${MUT_BUILD}/rank_correlation.o"
LINK_OBJS=""
for o in ${LIB_OBJS}; do
    case "$o" in
        */utils/rank_correlation.o) ;;
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

    # M7 needs a helper function that the replacement refers to.
    if [ "${mid}" = "M7" ]; then
        if ! add_d2_helper; then
            echo "${mid} MUT-NOOP  the helper could not be added -- harness defect"
            NOOP=$((NOOP + 1))
            continue
        fi
    fi

    # --- checksum: a mutation that did not apply is a harness defect.
    if cmp -s "${SRC}" "${BACKUP}"; then
        echo "${mid} MUT-NOOP  the source is unchanged -- harness defect"
        NOOP=$((NOOP + 1))
        continue
    fi

    # --- build. The mutated TU is compiled separately and swapped in.
    rm -f "${MUT_OBJ}" "${MUT_BUILD}/test_rank_correlation"
    if ! g++ -std=c++17 -O2 -Wall -Wextra -march=native -I"${REPO_ROOT}/include" \
            -c "${SRC}" -o "${MUT_OBJ}" 2> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the mutated TU did not compile -- a BUILD verdict, not a coverage one"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ -std=c++17 -O2 -I"${REPO_ROOT}/include" \
            -c "${TEST_SRC}" -o "${MUT_BUILD}/test_rank_correlation.o" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the test did not compile -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ "${MUT_BUILD}/test_rank_correlation.o" "${MUT_OBJ}" ${LINK_OBJS} \
            -o "${MUT_BUILD}/test_rank_correlation" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the link failed -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi

    # --- run.
    "${MUT_BUILD}/test_rank_correlation" > "${MUT_BUILD}/run.log" 2>&1
    rc=$?

    if [ $rc -eq 0 ]; then
        # The suite PASSED with the mutation in place. That is a hole in the
        # SUITE, and it is reported as such -- never as a pass.
        echo "${mid} SURVIVED  ${desc}"
        SURVIVED=$((SURVIVED + 1))
    elif [ $rc -ge 128 ]; then
        # A CRASH is a THIRD verdict class: the binary compiled and then
        # died, which says nothing about whether the suite covers the
        # behaviour.
        echo "${mid} MUT-CRASH  ${desc}  (rc=${rc})"
        CRASHED=$((CRASHED + 1))
    else
        n_fail=$(grep -c '\[FAIL\]' "${MUT_BUILD}/run.log")
        first=$(grep -m1 '\[FAIL\]' "${MUT_BUILD}/run.log" | cut -c1-120)
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
