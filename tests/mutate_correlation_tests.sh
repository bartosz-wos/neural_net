#!/usr/bin/env bash
# =============================================================================
# tests/mutate_correlation_tests.sh -- mutation harness for the correlation
# hypothesis tests: utils/correlation_tests.{h,cpp}.
#
# WHY THIS FILE EXISTS
# =============================================================================
#
# A suite is, by construction, unable to report a coverage gap: it can only
# report what it asserts. This harness breaks the implementation in specific
# ways and requires that the suite FAIL each time. It is the only thing in
# this repo that can find "the suite is green and the behaviour is wrong".
#
# The specifications are the oracle (tests/correlation_tests_oracle.py,
# float64 numpy, measured against scipy 1.17.1 and R 4.5.3) and the
# conventions C1-C8 in the header.
#
# =============================================================================
# THE MUTATIONS, AND WHAT EACH ONE IS FOR
# =============================================================================
#
#   M1  the t statistic's degrees of freedom are n-2 -> n
#       -> C1. The p-value depends on the dof through BOTH the t statistic
#          and the survival function, so this moves the answer.
#   M2  the t statistic's dof is n-2 -> n-1
#       -> C1, the other direction.
#   M3  the statistic is NOT clipped to [-1, 1]
#       -> C4. **THE MOST IMPORTANT MUTATION IN THIS HARNESS.** Without
#          the clip, `1 - r*r` is negative for the statistic this repo's
#          own `pearson_r` produces (measured at 1.0000000000000011102),
#          the square root is NaN, and the p-value of a PERFECT
#          correlation is NaN. The suite's 1.4a/1.4b checks exist for
#          exactly this.
#   M4  the clip is applied to the WRONG SIDE (the lower bound only)
#       -> C4, the other half.
#   M5  the zero-denominator case returns 1.0 instead of 0.0
#       -> C7. A PERFECT correlation has p-value 0.0, not 1.0.
#   M6  the n == 2 case returns 1.0 instead of NaN
#       -> C3. The references DISAGREE at n == 2, so the module ships NaN.
#   M7  the n == 2 case THROWS instead of returning NaN
#       -> C3, the other direction: a degenerate statistic is not an error.
#   M8  the Pearson p-value is the ONE-SIDED upper tail instead of the
#       two-sided one
#       -> C6. **THE SIGN OF THE STATISTIC IS UNOBSERVABLE** (the
#          formula computes `dof/(dof + at*at)`, and at*at is the same
#          for +t and -t), so the mutation that merely dropped the
#          absolute value was PROVABLY INERT. The observable difference
#          between one-sided and two-sided is the VALUE -- the one-sided
#          upper tail is HALF the two-sided one -- so the mutation
#          targets the VALUE.
#   M9  the Pearson and the Spearman p-values use DIFFERENT formulas
#       -> C2: they share ONE formula, so the suite's 2.3 loop (which
#          compares them at 17 statistics and 38 sample sizes) catches it.
#   M10 the Kendall variance DROPS the `2*xtie*ytie/m` term
#       -> C2. The tie terms are load-bearing, and the suite's 3.4 check
#          is the one that sees it.
#   M11 the Kendall variance DROPS the `x0*y0/(9*m*(n-2))` term
#       -> C2, the other tie term.
#   M12 the Kendall variance's degrees of freedom term is n-2 -> n
#       -> C2.
#   M13 the tie-GROUP statistics use the tie PAIR COUNT instead
#       -> C2. **THE SECOND MOST IMPORTANT MUTATION.** F5's x has the
#          groups {4,2} and F6's x has {3,3,2}: the SAME number of tied
#          pairs, DIFFERENT variances. A variance built from the pair
#          count alone gives the same answer for both, and the suite's
#          3.3c check sees it.
#   M14 the Kendall p-value uses `2*(1 - normal_cdf(|z|))` instead of the
#       erfc form
#       -> the CANCELLATION defect. `2*(1-Phi(|z|))` IS erfc(|z|/sqrt(2))
#          mathematically, but the subtraction form loses every digit in
#          the tail: MEASURED on the F3 fixture (z = 6.0346) it gives
#          1.5932055674738876e-09 against the exact 1.59320548535935e-09
#          -- a 5.2e-8 relative error. **This mutation was a REAL BUG in
#          this module's first draft, and the suite caught it.**
#   M15 the two-sided normal survival is multiplied by 2
#       -> C6. The erfc already carries both tails.
#   M16 the p-value is NOT clamped to [0, 1]
#       -> the clamp. **THE FIRST FORM OF THIS MUTATION WAS PROVABLY
#          INERT** on the shipped path, because `erfc` already returns a
#          value in [0, 1] -- so removing the clamp changed nothing. The
#          clamp is load-bearing for the OTHER route (the
#          `2*(1 - normal_cdf(|z|))` form, whose arithmetic can exceed 1
#          at the boundary), so the mutation now targets the CLAMP'S
#          EFFECT at the boundary: it replaces the clamped value with a
#          form that can exceed 1.
#   M17 the Pearson p-value's argument order is (n, r) instead of (r, n)
#       -> C8: the statistic comes first.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY PAID FOR
# =============================================================================
#
#   1. `restore()` must NEVER delete its backup. It is called once PER
#      mutation and once on EXIT, so a self-cleaning restore works exactly
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
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET. It legitimately
#      IS in the library's list (it is the TU being mutated), so the
#      mutated TU is compiled SEPARATELY, dropped from the library's copy,
#      and linked in its place. Excluding it without replacing it breaks
#      the link for the wrong reason.
#   4. `MUT-BROKEN` means "the build failed". It is a separate class and is
#      never counted as a pass.
#   5. A CRASH (rc >= 128) is a THIRD class, not a verdict.
#   6. A mutation that DID NOT APPLY is a HARNESS DEFECT, not a "SURVIVED"
#      verdict. The tree is checksummed before and after every edit, and
#      the anchor must occur EXACTLY ONCE.
#   7. NEVER run a build concurrently with this harness.
#   8. The pristine tree must be GREEN before any mutation runs.
#   9. The tree is rebuilt and re-verified ON EXIT before any verdict is
#      reported.
#
# =============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 99

SRC="include/nn/utils/correlation_tests.cpp"
TEST_SRC="tests/test_correlation_tests.cpp"
MUT_BUILD="${REPO_ROOT}/build/mutation/correlation_tests"
SUITE_NAME="test_correlation_tests"

STRICT=0
VERIFY_ONLY=0
POST_VERIFY=0
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
        --verify) VERIFY_ONLY=1 ;;
        -h|--help)
            echo "usage: $0 [--verify] [--strict]"
            echo "  --verify  check every anchor against the source and stop"
            echo "  --strict  treat the declared-inert survivors as failures"
            exit 0 ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 99 ;;
    esac
done

if [ ! -f "${SRC}" ]; then
    echo "HARNESS DEFECT: ${SRC} does not exist." >&2
    exit 99
fi

# -----------------------------------------------------------------------------
# The backup. restore() NEVER deletes it; cleanup does, on EXIT.
# -----------------------------------------------------------------------------
BAK="$(mktemp)"
cp "${SRC}" "${BAK}"
SCRATCH="$(mktemp -d)"
MUT_DIR="${SCRATCH}/muts"
mkdir -p "${MUT_DIR}" "${MUT_BUILD}"

restore() {
    # ATOMIC: cp to a temp then mv, so a concurrent reader never sees a
    # half-written source.
    cp "${BAK}" "${SRC}.tmp" 2>/dev/null && mv -f "${SRC}.tmp" "${SRC}"
}

cleanup() {
    local rc=$?
    restore
    rm -f "${BAK}"
    if [ "${VERIFY_ONLY}" -eq 0 ] && [ "${POST_VERIFY}" -eq 1 ]; then
        # THE POST-HARNESS REBUILD-AND-VERIFY DEFENCE. It rebuilds ONLY
        # the one test binary, out of the RESTORED source, and runs it.
        # It never invokes `make tests`.
        rm -f "${MUT_BUILD}/correlation_tests.o" "${MUT_BUILD}/${SUITE_NAME}"
        if build_suite; then
            if "${MUT_BUILD}/${SUITE_NAME}" >/dev/null 2>&1; then
                echo ""
                echo "POST-HARNESS: tree restored, rebuilt, and GREEN."
            else
                echo "" >&2
                echo "POST-HARNESS FAILURE: the restored tree does not pass. STOP." >&2
                rc=1
            fi
        else
            echo "" >&2
            echo "POST-HARNESS FAILURE: the restored tree does not BUILD. STOP." >&2
            rc=1
        fi
    fi
    rm -rf "${MUT_BUILD}" "${SCRATCH}"
    exit $rc
}
trap cleanup EXIT INT TERM

checksum() { md5sum "$1" | cut -d' ' -f1; }

# -----------------------------------------------------------------------------
# The link set, from make itself, with a count assertion.
# -----------------------------------------------------------------------------
mapfile -t ALL_OBJS < <(make print-lib-objs 2>/dev/null)
if [ "${#ALL_OBJS[@]}" -lt 10 ]; then
    echo "HARNESS DEFECT: the link set has only ${#ALL_OBJS[@]} objects (need >= 10)." >&2
    echo "REFUSING TO RUN -- every verdict below would be false." >&2
    exit 99
fi
LIB_LINK=""
for o in "${ALL_OBJS[@]}"; do
    case "$o" in
        */utils/correlation_tests.o|*/${SUITE_NAME}.o) ;;
        *) LIB_LINK="${LIB_LINK} $o" ;;
    esac
done
LIB_COUNT=$(echo "${LIB_LINK}" | wc -w)
if [ "${LIB_COUNT}" -lt 10 ]; then
    echo "HARNESS DEFECT: the link set collapsed to ${LIB_COUNT} objects." >&2
    exit 99
fi
echo "link set: ${LIB_COUNT} objects (the mutated TU is compiled separately and linked in)"

CXXFLAGS_LOCAL="-std=c++17 -O2 -Wall -Wextra -march=native -Iinclude"

build_suite() {
    rm -f "${MUT_BUILD}/correlation_tests.o" "${MUT_BUILD}/${SUITE_NAME}"
    g++ ${CXXFLAGS_LOCAL} -c "${SRC}" -o "${MUT_BUILD}/correlation_tests.o" \
        2> "${MUT_BUILD}/compile.log" || return 1
    g++ ${CXXFLAGS_LOCAL} -c "${TEST_SRC}" -o "${MUT_BUILD}/${SUITE_NAME}.o" \
        2>> "${MUT_BUILD}/compile.log" || return 1
    g++ "${MUT_BUILD}/${SUITE_NAME}.o" "${MUT_BUILD}/correlation_tests.o" ${LIB_LINK} \
        -o "${MUT_BUILD}/${SUITE_NAME}" 2>> "${MUT_BUILD}/compile.log" || return 1
    return 0
}

# -----------------------------------------------------------------------------
# The mutation spec, and the splitter that turns it into one OLD file and one
# NEW file per mutation.
#
#   @@MUTATION <id>
#   @@DESCRIPTION <one line>
#   @@EXPECTATION caught | inert
#   @@WHY <why an inert mutation is inert>
#   @@OLD
#   ...the exact block in the source...
#   @@NEW
#   ...the exact replacement...
# -----------------------------------------------------------------------------
read -r -d '' MUT_SPEC <<'MUTSPEC'
@@MUTATION M1
@@DESCRIPTION the t statistic's degrees of freedom are n-2 -> n
@@EXPECTATION caught
@@OLD
    const double dof = static_cast<double>(n) - 2.0;
@@NEW
    const double dof = static_cast<double>(n) - 1.0;
@@MUTATION M2
@@DESCRIPTION the t statistic's degrees of freedom are n-2 -> n-1
@@EXPECTATION caught
@@OLD
    const double dof = static_cast<double>(n) - 2.0;
@@NEW
    const double dof = static_cast<double>(n) - 3.0;
@@MUTATION M3
@@DESCRIPTION the statistic is not clipped to [-1, 1]
@@EXPECTATION caught
@@OLD
double clip_statistic(double r) {
    if (r > 1.0) return 1.0;
    if (r < -1.0) return -1.0;
    return r;
}
@@NEW
double clip_statistic(double r) {
    return r;
}
@@MUTATION M4
@@DESCRIPTION the clip is applied to the lower bound only
@@EXPECTATION caught
@@OLD
double clip_statistic(double r) {
    if (r > 1.0) return 1.0;
    if (r < -1.0) return -1.0;
    return r;
}
@@NEW
double clip_statistic(double r) {
    if (r < -1.0) return -1.0;
    return r;
}
@@MUTATION M5
@@DESCRIPTION the p-value at a perfect correlation is 1.0 instead of 0.0
@@EXPECTATION caught
@@OLD
    const double denom = 1.0 - rc * rc;
@@NEW
    const double denom = 1.0 - rc * rc;
    if (denom <= 0.0) return 1.0;
@@MUTATION M6
@@DESCRIPTION the n == 2 case returns 1.0 instead of NaN
@@EXPECTATION caught
@@OLD
    if (n == 2) return std::numeric_limits<double>::quiet_NaN();
    // C5: an undefined correlation has an undefined p-value.
@@NEW
    if (n == 2) return 1.0;
    // C5: an undefined correlation has an undefined p-value.
@@MUTATION M7
@@DESCRIPTION the n == 2 case throws instead of returning NaN
@@EXPECTATION caught
@@OLD
    if (n == 2) return std::numeric_limits<double>::quiet_NaN();
    // C5: an undefined correlation has an undefined p-value.
@@NEW
    if (n == 2) throw std::invalid_argument("pearson_p_value: n == 2");
    // C5: an undefined correlation has an undefined p-value.
@@MUTATION M8
@@DESCRIPTION the Pearson p-value is the one-sided upper tail instead of the two-sided one
@@EXPECTATION caught
@@OLD
    const double t = rc * std::sqrt(dof / denom);
    return student_t_two_sided_survival(t, dof);
@@NEW
    const double t = rc * std::sqrt(dof / denom);
    return 0.5 * student_t_two_sided_survival(t, dof);
@@MUTATION M9
@@DESCRIPTION the Spearman p-value uses a different formula from the Pearson one
@@EXPECTATION caught
@@OLD
double spearman_p_value(double rho, size_t n) {
    return correlation_p_value(rho, n, "spearman_p_value");
}
@@NEW
double spearman_p_value(double rho, size_t n) {
    const double dof = static_cast<double>(n) - 2.0;
    const double denom = 1.0 - rho * rho;
    if (denom <= 0.0) return 0.0;
    const double t = std::fabs(rho) * std::sqrt(dof / denom);
    return student_t_two_sided_survival(t, dof - 1.0);
}
@@MUTATION M10
@@DESCRIPTION the Kendall variance drops the 2*xtie*ytie/m term
@@EXPECTATION caught
@@OLD
    const double variance =
        (m * (2.0 * dn + 5.0) - sx.x1 - sy.x1) / 18.0 +
        (2.0 * sx.pairs * sy.pairs) / m +
        (sx.x0 * sy.x0) / (9.0 * m * (dn - 2.0));
@@NEW
    const double variance =
        (m * (2.0 * dn + 5.0) - sx.x1 - sy.x1) / 18.0 +
        (sx.x0 * sy.x0) / (9.0 * m * (dn - 2.0));
@@MUTATION M11
@@DESCRIPTION the Kendall variance drops the x0*y0/(9*m*(n-2)) term
@@EXPECTATION caught
@@OLD
    const double variance =
        (m * (2.0 * dn + 5.0) - sx.x1 - sy.x1) / 18.0 +
        (2.0 * sx.pairs * sy.pairs) / m +
        (sx.x0 * sy.x0) / (9.0 * m * (dn - 2.0));
@@NEW
    const double variance =
        (m * (2.0 * dn + 5.0) - sx.x1 - sy.x1) / 18.0 +
        (2.0 * sx.pairs * sy.pairs) / m;
@@MUTATION M12
@@DESCRIPTION the Kendall variance's degrees of freedom term uses n instead of n-2
@@EXPECTATION caught
@@OLD
        (sx.x0 * sy.x0) / (9.0 * m * (dn - 2.0));
@@NEW
        (sx.x0 * sy.x0) / (9.0 * m * dn);
@@MUTATION M13
@@DESCRIPTION the tie-group statistics use the tie PAIR COUNT instead
@@EXPECTATION caught
@@OLD
        if (c >= 2.0) {
            out.pairs += c * (c - 1.0) / 2.0;
            out.x0 += c * (c - 1.0) * (c - 2.0);
            out.x1 += c * (c - 1.0) * (2.0 * c + 5.0);
        }
@@NEW
        if (c >= 2.0) {
            out.pairs += c * (c - 1.0) / 2.0;
            out.x0 += out.pairs * (c - 2.0);
            out.x1 += out.pairs * (2.0 * c + 5.0);
        }
@@MUTATION M14
@@DESCRIPTION the Kendall p-value uses 2*(1 - normal_cdf(|z|)) instead of the erfc form
@@EXPECTATION caught
@@OLD
    const double p = std::erfc(std::fabs(z) / std::sqrt(2.0));
    return std::min(1.0, std::max(0.0, p));
@@NEW
    const double p = 2.0 * (1.0 - ::normal_cdf(std::fabs(z)));
    return std::min(1.0, std::max(0.0, p));
@@MUTATION M15
@@DESCRIPTION the two-sided normal survival is multiplied by 2
@@EXPECTATION caught
@@OLD
    const double p = std::erfc(std::fabs(z) / std::sqrt(2.0));
@@NEW
    const double p = 2.0 * std::erfc(std::fabs(z) / std::sqrt(2.0));
@@MUTATION M16
@@DESCRIPTION the Kendall p-value is not clamped to [0, 1]
@@EXPECTATION caught
@@OLD
    const double p = std::erfc(std::fabs(z) / std::sqrt(2.0));
    return std::min(1.0, std::max(0.0, p));
@@NEW
    const double p = std::erfc(std::fabs(z) / std::sqrt(2.0));
    return p * 1.0000001;
@@MUTATION M17
@@DESCRIPTION the Pearson p-value's argument order is (n, r) instead of (r, n)
@@EXPECTATION caught
@@OLD
double pearson_p_value(double r, size_t n) {
    return correlation_p_value(r, n, "pearson_p_value");
}
@@NEW
double pearson_p_value(double r, size_t n) {
    return correlation_p_value(n, r, "pearson_p_value");
}
MUTSPEC

# -----------------------------------------------------------------------------
# Split the spec into one OLD file and one NEW file per mutation.
# -----------------------------------------------------------------------------
python3 - "${MUT_SPEC}" "${MUT_DIR}" <<'PYEOF'
import os, sys
spec, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
blocks = []
cur = None
mode = None
for line in spec.split("\n"):
    if line.startswith("@@MUTATION "):
        cur = {"id": line.split(None, 1)[1].strip(), "old": [], "new": []}
        blocks.append(cur)
        mode = None
        continue
    if cur is None:
        continue
    if line.startswith("@@OLD"):
        mode = "old"; continue
    if line.startswith("@@NEW"):
        mode = "new"; continue
    if line.startswith("@@"):
        mode = None; continue
    if mode:
        cur[mode].append(line)
for b in blocks:
    with open(os.path.join(outdir, b["id"] + ".old"), "w") as f:
        f.write("\n".join(b["old"]) + "\n")
    with open(os.path.join(outdir, b["id"] + ".new"), "w") as f:
        f.write("\n".join(b["new"]) + "\n")
print(len(blocks))
PYEOF
N_MUT=$(ls "${MUT_DIR}"/*.old 2>/dev/null | wc -l)
echo "mutations: ${N_MUT}"

# -----------------------------------------------------------------------------
# The PRISTINE tree must be GREEN before any mutation runs. A harness that
# starts from a red tree reports garbage.
# -----------------------------------------------------------------------------
if ! build_suite; then
    echo "HARNESS DEFECT: the pristine suite does not BUILD. STOP." >&2
    tail -20 "${MUT_BUILD}/compile.log" >&2
    exit 99
fi
if ! "${MUT_BUILD}/${SUITE_NAME}" > "${MUT_BUILD}/pristine.log" 2>&1; then
    echo "HARNESS DEFECT: the pristine suite does not PASS. STOP." >&2
    tail -20 "${MUT_BUILD}/pristine.log" >&2
    exit 99
fi
echo "pristine suite: GREEN"

if [ "${VERIFY_ONLY}" -eq 1 ]; then
    for old_file in "${MUT_DIR}"/*.old; do
        mid="$(basename "${old_file}" .old)"
        count=$(python3 - "${SRC}" "${old_file}" <<'PYEOF'
import sys
src = open(sys.argv[1]).read()
old = open(sys.argv[2]).read()
print(src.count(old.rstrip("\n")))
PYEOF
)
        if [ "${count}" != "1" ]; then
            echo "${mid} MUT-ANCHOR  the anchor occurs ${count} times (need exactly 1)"
        else
            echo "${mid} anchor ok"
        fi
    done
    exit 0
fi

# -----------------------------------------------------------------------------
# The main loop.
# -----------------------------------------------------------------------------
CAUGHT=0
SURVIVED=0
BROKEN=0
CRASHED=0
NOOP=0
ANCHOR_BAD=0

for old_file in "${MUT_DIR}"/*.old; do
    mid="$(basename "${old_file}" .old)"
    new_file="${MUT_DIR}/${mid}.new"

    # --- restore FIRST. Every iteration starts from the pristine source.
    restore

    before="$(checksum "${SRC}")"

    # --- apply the mutation with python (no shell delimiter hazards).
    python3 - "${SRC}" "${old_file}" "${new_file}" <<'PYEOF'
import sys
path, oldf, newf = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
old = open(oldf).read().rstrip("\n")
new = open(newf).read().rstrip("\n")
count = src.count(old)
if count != 1:
    sys.stderr.write("ANCHOR count=%d\n" % count)
    sys.exit(3)
open(path, "w").write(src.replace(old, new))
PYEOF
    rc_edit=$?
    if [ ${rc_edit} -eq 3 ]; then
        echo "${mid} MUT-ANCHOR  the anchor is not unique -- harness defect, not a verdict"
        ANCHOR_BAD=$((ANCHOR_BAD + 1))
        continue
    elif [ ${rc_edit} -ne 0 ]; then
        echo "${mid} MUT-NOOP  the edit failed -- harness defect, not a verdict"
        NOOP=$((NOOP + 1))
        continue
    fi

    after="$(checksum "${SRC}")"
    if [ "${before}" = "${after}" ]; then
        echo "${mid} MUT-NOOP  the source's checksum did not change -- harness defect"
        NOOP=$((NOOP + 1))
        continue
    fi

    # --- build. The mutated TU is compiled separately and swapped in.
    if ! build_suite; then
        echo "${mid} MUT-BROKEN  the mutated TU did not build -- a BUILD verdict, not a coverage one"
        BROKEN=$((BROKEN + 1))
        continue
    fi

    # --- run.
    "${MUT_BUILD}/${SUITE_NAME}" > "${MUT_BUILD}/${mid}.log" 2>&1
    rc=$?
    if [ ${rc} -ge 128 ]; then
        # A CRASH IS NOT A VERDICT. It says nothing about whether the suite
        # covers the behaviour.
        echo "${mid} MUT-CRASH  the binary died (rc=${rc}) -- a crash is not a verdict"
        CRASHED=$((CRASHED + 1))
        continue
    fi
    if [ ${rc} -ne 0 ]; then
        nfail=$(grep -c '\[FAIL\]' "${MUT_BUILD}/${mid}.log" || true)
        echo "${mid} CAUGHT  the suite FAILED (${nfail} failing checks)"
        grep -m 3 '\[FAIL\]' "${MUT_BUILD}/${mid}.log" | sed 's/^/                 /'
        CAUGHT=$((CAUGHT + 1))
    else
        echo "${mid} SURVIVED  the suite stayed GREEN -- a COVERAGE GAP"
        SURVIVED=$((SURVIVED + 1))
    fi
done

# -----------------------------------------------------------------------------
# The report.
# -----------------------------------------------------------------------------
restore
POST_VERIFY=1

echo ""
echo "FINAL: ${CAUGHT} caught / ${SURVIVED} survived / ${BROKEN} broken / ${CRASHED} crashed / ${NOOP} noop / ${ANCHOR_BAD} anchor"

if [ "${SURVIVED}" -ne 0 ]; then
    echo ""
    echo "A SURVIVOR IS A SPECIFICATION FOR A BETTER CHECK, NOT A NUDGE TO DROP"
    echo "THE MUTATION. Each one above names a behaviour the suite does not"
    echo "currently assert."
fi
if [ "${STRICT}" -eq 1 ] && [ "${SURVIVED}" -ne 0 ]; then
    exit 1
fi
exit 0
