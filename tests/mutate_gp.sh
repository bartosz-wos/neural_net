#!/usr/bin/env bash
# tests/mutate_gp.sh -- mutation harness for utils/gaussian_process.{h,cpp}.
#
# This harness exists because the repo has recorded TWENTY-THREE instances of a
# mutation harness reporting a confident WRONG verdict. Every defect below is
# one this repo has already paid for, so they are addressed BY CONSTRUCTION
# rather than by memory:
#
#   1. restore() MUST NOT DELETE ITS OWN BACKUP. It is called once per
#      mutation AND on EXIT; a restore that `rm`s the backup works exactly
#      once, every later restore is a silent no-op printing `cp: cannot stat`,
#      and the tree is left holding whatever mutant ran last. Cleanup happens
#      exactly once, on EXIT, and restore() is ATOMIC (`cp tmp && mv -f`).
#   2. NEVER RECOVER THE LINK SET BY REGEX. `LIB_OBJS := $(LIB_SRCS:...)` is a
#      RECURSIVE make variable whose literal text contains no .o filenames; a
#      regex against it yields ZERO objects and turns every mutation into a
#      false "did not link". This harness asks make, and ASSERTS the count.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET, or every mutation
#      double-defines every symbol and reports CAUGHT (compile error). A link
#      error is not a mutation verdict. The mutated TU is compiled SEPARATELY
#      and swapped in for the library object.
#   4. `MUT-BROKEN` MEANS THE BUILD FAILED. It is reported separately from
#      CAUGHT and never counted as a pass.
#   5. A MUTATION THAT DID NOT APPLY IS A HARNESS DEFECT, not a verdict. The
#      tree is checksummed before and after every edit.
#   6. NO EMBEDDED NEWLINES AND NO DELIMITED SCRIPT. The edit is done by a
#      Python editor with `str.count`/`str.replace`, which has no delimiter
#      concept -- C++ is full of `/`, `|`, `&` and `\`.
#   7. A MISSING BINARY IS `UNAVAILABLE`, never a verdict.
#   8. The spec loop refuses any entry that does not name a known file, so a
#      stray marker can never be read as an editor script.
#   9. A MUTATION THAT CRASHES (rc >= 128) IS NOT A VERDICT. It is a third
#      verdict class: the binary compiled, then died.
#  10. `rm -f` the intermediates before every build -- a stale .o from the
#      previous mutation reproduces a build failure that does not exist alone.
#  11. THE ANCHOR MUST BE UNIQUE, and the uniqueness count comes from the
#      SAME engine that performs the edit (`str.count`), not from `grep -c`
#      (which counts MATCHING LINES, not matches).
#  12. AFTER ADDING A FEATURE, RE-CHECK EVERY EXISTING MUTATION'S ANCHOR
#      COUNT. A new feature that reuses a line invalidates an old anchor
#      silently.
#  13. NO BACKTICKS, NO `$(...)`, NO `!` in double-quoted strings. A backtick
#      in a bash double-quoted string is COMMAND SUBSTITUTION, and it
#      silently truncates the description.
#
# NEVER RUN THIS HARNESS CONCURRENTLY with tests/mutate_crf.sh or any other
# harness that touches the SAME build/*.o files.

set -u
cd "$(dirname "$0")/.." || exit 1

SRC="include/nn/utils/gaussian_process.cpp"
HDR="include/nn/utils/gaussian_process.h"
TEST="tests/test_gp.cpp"
BIN="build/test_gp"
BAK_SRC=$(mktemp); BAK_HDR=$(mktemp)
BAK_MARK=".mutate_gp_tree_pristine"
cp "$SRC" "$BAK_SRC"; cp "$HDR" "$BAK_HDR"
cp "$SRC" "$BAK_MARK"

restore() {                      # atomic; NEVER deletes the backups
    cp "$BAK_SRC" "$SRC.tmp" 2>/dev/null && mv -f "$SRC.tmp" "$SRC"
    cp "$BAK_HDR" "$HDR.tmp" 2>/dev/null && mv -f "$HDR.tmp" "$HDR"
}
cleanup() { restore; rm -f "$BAK_SRC" "$BAK_HDR" "$BAK_MARK"; }
trap cleanup EXIT INT TERM

# ---- link set, from make, with a count assertion (defect 2) ----------------
LIB_LINK=$(make -s print-lib-objs 2>/dev/null | grep '\.o$' | grep -v '/test_' | tr '\n' ' ')
LIB_COUNT=$(echo "$LIB_LINK" | wc -w)
if [ "$LIB_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: recovered only $LIB_COUNT library objects (< 10)." >&2
    echo "Every verdict below would be false. REFUSING TO RUN." >&2
    exit 2
fi
MUT_OBJ="build/utils/gaussian_process.o"
case " $LIB_LINK " in
    *" $MUT_OBJ "*)
        LIB_LINK=$(echo "$LIB_LINK" | tr ' ' '\n' | grep -v "^${MUT_OBJ}\$" | tr '\n' ' ')
        echo "link set: $LIB_COUNT objects, minus $MUT_OBJ (compiled separately)"
        ;;
    *)
        echo "HARNESS DEFECT: $MUT_OBJ is not in the library link set at all." >&2
        exit 2 ;;
esac
LIB_LINK_COUNT=$(echo "$LIB_LINK" | wc -w)
if [ "$LIB_LINK_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: link set collapsed to $LIB_LINK_COUNT objects." >&2
    exit 2
fi

CXXFLAGS_LOCAL="-std=c++17 -O2 -Wall -Wextra -march=native -Iinclude"

build_and_run() {   # defect 10
    rm -f build/test_gp.o "$BIN" "$MUT_OBJ"
    make -s build/utils/gaussian_process.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$TEST" -o build/test_gp.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$SRC" -o "$MUT_OBJ" >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL $LIB_LINK "$MUT_OBJ" build/test_gp.o -o "$BIN" >/dev/null 2>&1 || return 1
    return 0
}

# ---- pristine self-check: the suite must be GREEN before any mutation ------
build_and_run
PRISTINE=$("$BIN" 2>&1 | sed -n 's/^=== SUMMARY: \([0-9]*\) passed.*/\1/p' | tail -1)
if [ "${PRISTINE:-0}" -eq 0 ] || ! "$BIN" >/dev/null 2>&1; then
    echo "HARNESS DEFECT: pristine tree is not green ($PRISTINE passed)." >&2
    echo "REFUSING TO RUN -- this is exactly how a leaked mutant is caught." >&2
    exit 2
fi
echo "pristine: $PRISTINE passed, 0 failed -- mutation may begin"

CAUGHT=0; SURVIVED=0; BROKEN=0; NOOP=0; UNAVAIL=0
declare -a TABLE

run_mutation() {   # $1=id  $2=file  $3=from  $4=to
    local id="$1" file="$2" from="$3" to="$4"
    restore

    if [ ! -f "$file" ]; then
        TABLE+=("$id|$file does not exist|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi
    # defect 11: the count comes from the same engine that performs the edit.
    local n
    n=$(python3 -c 'import sys; print(open(sys.argv[1],encoding="utf8").read().count(sys.argv[2]))' "$file" "$from")
    if [ "$n" -ne 1 ]; then
        TABLE+=("$id|anchor count = $n|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi

    local before after
    before=$(md5sum "$file" | cut -d' ' -f1)
    python3 - "$file" "$from" "$to" <<'PYEOF' || { restore; TABLE+=("$id|editor failed|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return; }
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding='utf8').read()
n = s.count(frm)
if n != 1:
    sys.stderr.write("anchor count %d\n" % n)
    sys.exit(3)
open(path, 'w', encoding='utf8').write(s.replace(frm, to))
PYEOF
    after=$(md5sum "$file" | cut -d' ' -f1)

    if [ "$before" = "$after" ]; then      # defect 5
        restore
        TABLE+=("$id|edit did not apply|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi

    build_and_run
    if [ ! -x "$BIN" ]; then               # defect 7
        restore
        TABLE+=("$id|no binary produced|UNAVAILABLE"); UNAVAIL=$((UNAVAIL+1)); return
    fi

    local out rc
    out=$("$BIN" 2>&1); rc=$?
    local np nf
    np=$(echo "$out" | sed -n 's/^=== SUMMARY: \([0-9]*\) passed.*/\1/p' | tail -1)
    nf=$(echo "$out" | sed -n 's/.*, \([0-9]*\) failed ===.*/\1/p' | tail -1)
    np=${np:-0}; nf=${nf:-0}

    if [ "$rc" -ge 128 ]; then              # defect 9: a crash is not a verdict
        restore
        TABLE+=("$id|CRASH rc=$rc|CRASH"); BROKEN=$((BROKEN+1)); return
    fi
    if [ "$nf" -gt 0 ]; then
        restore
        TABLE+=("$id|$np passed, $nf failed|CAUGHT"); CAUGHT=$((CAUGHT+1)); return
    fi
    restore
    TABLE+=("$id|suite still green|SURVIVED"); SURVIVED=$((SURVIVED+1))
}

echo
echo "=== gaussian process mutations ==="

# --- THE COMPLEXITY PENALTY: THE FACTOR OF TWO ---------------------------
# M01 halves the log-determinant term. The Cholesky form of -1/2 log|Ky| is
#      -sum(log L_ii) because log|Ky| = 2 sum(log L_ii). Halving it is a
#      factor-of-two error; the oracle measured a difference of 0.507131 on
#      the G2 fixture.
run_mutation M01 "$SRC" \
    "    return -0.5 * quad - logdet_half -" \
    "    return -0.5 * quad - 0.5 * logdet_half -"

# --- THE TWO TRIANGULAR SOLVES ---------------------------------------------
# M02 drops the second solve, so alpha = L^-1 y instead of L^-T L^-1 y. This
#      is the classic "one solve instead of two" bug and it changes the
#      predictive mean AND the log marginal likelihood.
run_mutation M02 "$SRC" \
    "    Tensor alpha = y.clone();
    forward_solve(L, alpha);
    backward_solve(L, alpha);

    double quad = 0.0;" \
    "    Tensor alpha = y.clone();
    forward_solve(L, alpha);

    double quad = 0.0;"

# M03 drops the FIRST solve instead -- alpha = L^T y, which is the transpose
#      of the right answer and produces a plausible-looking wrong number.
run_mutation M03 "$SRC" \
    "    Tensor alpha = y.clone();
    forward_solve(L, alpha);
    backward_solve(L, alpha);

    double quad = 0.0;" \
    "    Tensor alpha = y.clone();
    backward_solve(L, alpha);

    double quad = 0.0;"

# --- THE NOISE TERM --------------------------------------------------------
# M04 uses sn instead of sn^2 in the covariance. The noise is a VARIANCE, so
#      it enters as sn^2 I.
run_mutation M04 "$SRC" \
    "    const double sn2 = sigma_n * sigma_n;
    for (size_t i = 0; i < Ky.rows; ++i) Ky(i, i) += sn2;" \
    "    const double sn2 = sigma_n;
    for (size_t i = 0; i < Ky.rows; ++i) Ky(i, i) += sn2;"

# M05 drops the noise from the covariance entirely. Then a noise-free GP is
#      fitted to noisy data, and the interpolation property appears where it
#      should not.
run_mutation M05 "$SRC" \
    "    const double sn2 = sigma_n * sigma_n;
    for (size_t i = 0; i < Ky.rows; ++i) Ky(i, i) += sn2;" \
    "    const double sn2 = 0.0;
    for (size_t i = 0; i < Ky.rows; ++i) Ky(i, i) += sn2;"

# --- THE PREDICTIVE VARIANCE -----------------------------------------------
# M06 drops the -v^T v correction, so the predictive variance is just
#      k(x*,x*). The oracle's property 3.3 says the variance at a training
#      point must be ~0 when sn = 0 -- without the correction it is sf^2.
run_mutation M06 "$SRC" \
    "        out.variance(j, 0) = Kss(j, 0) - vv;" \
    "        out.variance(j, 0) = Kss(j, 0) + vv;"

# M07 drops the correction entirely.
run_mutation M07 "$SRC" \
    "        out.variance(j, 0) = Kss(j, 0) - vv;" \
    "        out.variance(j, 0) = Kss(j, 0);"

# --- THE CONSTANT TERM -----------------------------------------------------
# M08 drops the -n/2 log(2 pi) constant. The oracle measured a difference of
#      5.513631 for n = 6.
run_mutation M08 "$SRC" \
    "    return -0.5 * quad - logdet_half -
           0.5 * static_cast<double>(n) * std::log(2.0 * M_PI);" \
    "    return -0.5 * quad - logdet_half;"

# --- THE KERNEL PARAMETERISATION ------------------------------------------
# M09 divides the SE exponent by l instead of l^2. At r = 2, l = 2 that is
#      exp(-1) = 0.367879 instead of exp(-0.5) = 0.606531.
run_mutation M09 "$SRC" \
    "            return std::exp(-0.5 * r2 / (lengthscale * lengthscale));" \
    "            return std::exp(-0.5 * r2 / lengthscale);"

# M10 uses sf instead of sf^2 as the amplitude.
run_mutation M10 "$SRC" \
    "    const double sf2 = sigma_f * sigma_f;
    for (size_t i = 0; i < X.rows; ++i)
        for (size_t j = 0; j < X2.rows; ++j)
            K(i, j) = sf2 * kernel_from_r2(D(i, j), lengthscale, kernel);" \
    "    const double sf2 = sigma_f;
    for (size_t i = 0; i < X.rows; ++i)
        for (size_t j = 0; j < X2.rows; ++j)
            K(i, j) = sf2 * kernel_from_r2(D(i, j), lengthscale, kernel);"

# --- THE HYPERPARAMETER GRADIENT: THE HALF AND THE TRACE ------------------
# M11 drops the 1/2 in eq. (5.9). The oracle measured -8.028646365 against
#      -4.014323183 on the first fixture.
run_mutation M11 "$SRC" \
    "    g.d_lengthscale = 0.5 * trace_product(dK_l);" \
    "    g.d_lengthscale = trace_product(dK_l);"

# M12 drops the -Ky^-1 term from M = alpha alpha^T - Ky^-1.
run_mutation M12 "$SRC" \
    "            M(i, j) = alpha(i, 0) * alpha(j, 0) - Kinv(i, j);" \
    "            M(i, j) = alpha(i, 0) * alpha(j, 0);"

# M13 computes the trace as the sum of ALL ENTRIES OF THE PRODUCT instead of
#      the trace itself.
#
#      **THE FIRST DRAFT OF THIS MUTATION USED THE ELEMENTWISE SUM, AND IT
#      SURVIVED -- EXACTLY AS THE ORACLE PREDICTED.** For symmetric M and dK
#      (which they always are here: M = alpha alpha^T - Ky^-1 is symmetric
#      because Ky^-1 is, and dK/dtheta is symmetric because the kernel is),
#          trace(M dK) == sum_ij M_ij dK_ij   EXACTLY
#      and the oracle MEASURED it: -8.028646365323730 against
#      -8.028646365323732. **THE MUTATION WAS INERT -- it changed nothing,
#      not even in the last bit.** The suite was right to stay green.
#
#      The observable difference is the sum of the entries of the MATRIX
#      PRODUCT, sum_ij (M dK)_ij = -5.0094916378122845, which is a different
#      quantity. THIS mutation is written against the matrix product.
#
#      This is the repo's rule made concrete: **a mutation that provably
#      changes nothing is not a vacuous test** -- the honest verdict is
#      "inert", not "coverage gap". The oracle caught this one BEFORE the
#      harness ran.
run_mutation M13 "$SRC" \
    "        double s = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) s += M(i, j) * dK(j, i);
        return s;" \
    "        double s = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                double p = 0.0;
                for (size_t k = 0; k < n; ++k) p += M(i, k) * dK(k, j);
                s += p;
            }
        return s;"

# --- THE LEAVE-ONE-OUT QUANTITIES ------------------------------------------
# M14 uses the NOISE-FREE kernel matrix instead of the noisy one. The oracle
#      resolved this by exact refit: the noisy reading agrees to 3.331e-16
#      and the noise-free reading differs by 1.007476.
run_mutation M14 "$SRC" \
    "    Tensor Ky = kernel_matrix_impl(X, X, lengthscale, sigma_f, kernel);
    const double sn2 = sigma_n * sigma_n;
    for (size_t i = 0; i < n; ++i) Ky(i, i) += sn2;
    Tensor L = Ky.clone();
    if (!cholesky_in_place(L)) {
        throw std::invalid_argument(
            \"gp: the covariance matrix is not positive definite (the \"
            \"Cholesky factorisation failed)\");
    }

    // Ky^-1, from the triangular solves." \
    "    Tensor Ky = kernel_matrix_impl(X, X, lengthscale, sigma_f, kernel);
    Tensor L = Ky.clone();
    if (!cholesky_in_place(L)) {
        throw std::invalid_argument(
            \"gp: the covariance matrix is not positive definite (the \"
            \"Cholesky factorisation failed)\");
    }

    // Ky^-1, from the triangular solves."

# M15 drops the y_i term from mu_i = y_i - [Ky^-1 y]_i / [Ky^-1]_ii. The
#      book's own remark -- "the mean mu_i is in fact independent of y_i" --
#      is exactly what this breaks, because the y_i in the subtraction is
#      what cancels it.
run_mutation M15 "$SRC" \
    "        out.mean(i, 0) = y(i, 0) - alpha(i, 0) / Kinv(i, i);" \
    "        out.mean(i, 0) = -alpha(i, 0) / Kinv(i, i);"

# M16 inverts the LOO variance.
run_mutation M16 "$SRC" \
    "        out.variance(i, 0) = 1.0 / Kinv(i, i);" \
    "        out.variance(i, 0) = Kinv(i, i);"

# --- THE SINGULARITY DETECTION ---------------------------------------------
# M17 removes the Cholesky failure detection in `factorize_ky`, so a singular
#      Ky silently produces nan instead of throwing. The oracle's property
#      6.1 requires the DETECTION.
#
#      **THE FIRST DRAFT OF THIS MUTATION ANCHORED ON `cholesky_in_place(L)`,
#      WHICH APPEARS IN THE TWO INLINE FACTORISATIONS BUT NOT IN
#      `factorize_ky` -- so the anchor count was 0 and the harness reported a
#      HARNESS DEFECT rather than a verdict.** That is the harness working:
#      an anchor that does not apply is refused, not silently skipped.
run_mutation M17 "$SRC" \
    "    if (!cholesky_in_place(Ky)) {
        throw std::invalid_argument(
            \"gp: the covariance matrix is not positive definite (the \"
            \"Cholesky factorisation failed)\");
    }
    return Ky;" \
    "    cholesky_in_place(Ky);
    return Ky;"

# --- THE NOISY-TARGET PREDICTIVE VARIANCE ---------------------------------
# M18 drops the sn^2 addition in gp_predict_noisy, so the noisy-target
#      variance equals the latent variance.
run_mutation M18 "$SRC" \
    "    const double sn2 = sigma_n * sigma_n;
    for (size_t i = 0; i < p.variance.rows; ++i) p.variance(i, 0) += sn2;" \
    "    const double sn2 = 0.0;
    for (size_t i = 0; i < p.variance.rows; ++i) p.variance(i, 0) += sn2;"

# --- THE GUARDS -------------------------------------------------------------
# M19 removes the length-scale guard.
run_mutation M19 "$SRC" \
    "    if (!std::isfinite(lengthscale) || lengthscale <= 0.0) {
        throw std::invalid_argument(
            \"gp: the length-scale must be finite and strictly positive\");
    }" \
    "    if (false) {
        throw std::invalid_argument(
            \"gp: the length-scale must be finite and strictly positive\");
    }"

# M20 removes the signal-variance guard.
run_mutation M20 "$SRC" \
    "    if (!std::isfinite(sigma_f) || sigma_f <= 0.0) {
        throw std::invalid_argument(
            \"gp: the signal variance must be finite and strictly positive\");
    }" \
    "    if (false) {
        throw std::invalid_argument(
            \"gp: the signal variance must be finite and strictly positive\");
    }"

# M21 removes the noise guard, so a negative noise variance is accepted and
#      silently adds a NEGATIVE variance to the diagonal.
run_mutation M21 "$SRC" \
    "    if (has_noise && (!std::isfinite(sigma_n) || sigma_n < 0.0)) {
        throw std::invalid_argument(
            \"gp: the noise variance must be finite and non-negative\");
    }" \
    "    if (false) {
        throw std::invalid_argument(
            \"gp: the noise variance must be finite and non-negative\");
    }"

# M22 removes the target-shape guard.
run_mutation M22 "$SRC" \
    "    if (y.rows != X.rows || y.cols != 1) {" \
    "    if (false) {"

# M23 removes the unknown-kernel guard.
run_mutation M23 "$SRC" \
    "    if (kernel != GP_KERNEL_SE && kernel != GP_KERNEL_MATERN32 &&
        kernel != GP_KERNEL_MATERN52) {
        throw std::invalid_argument(\"gp: unknown kernel type\");
    }" \
    "    if (false) {
        throw std::invalid_argument(\"gp: unknown kernel type\");
    }"

# --- THE MATERN KERNELS ----------------------------------------------------
# M24 makes the Matern 3/2 kernel identical to the SE kernel, so the suite's
#      "the three kernels are different functions" check fires.
run_mutation M24 "$SRC" \
    "            const double a = std::sqrt(3.0) * r / lengthscale;
            return (1.0 + a) * std::exp(-a);" \
    "            const double a = std::sqrt(3.0) * r / lengthscale;
            return std::exp(-a);"

# M25 drops the polynomial factor from the Matern 5/2 kernel.
run_mutation M25 "$SRC" \
    "            return (1.0 + a + 5.0 * r2 / (3.0 * lengthscale * lengthscale)) *
                   std::exp(-a);" \
    "            return (1.0 + a) * std::exp(-a);"

echo
printf '%-6s %-52s %s\n' "ID" "DETAIL" "VERDICT"
printf '%-6s %-52s %s\n' "----" "-----" "-------"
for row in "${TABLE[@]}"; do
    IFS='|' read -r a b c <<< "$row"
    printf '%-6s %-52s %s\n' "$a" "$b" "$c"
done

echo
echo "=== caught $CAUGHT / survived $SURVIVED / broken $BROKEN / noop $NOOP / unavailable $UNAVAIL ==="

# ---- post-harness verification: the tree must be pristine again -----------
restore
build_and_run
FINAL=$("$BIN" 2>&1 | sed -n 's/^=== SUMMARY: \([0-9]*\) passed.*/\1/p' | tail -1)
if "$BIN" >/dev/null 2>&1 && [ "${FINAL:-0}" -eq "$PRISTINE" ]; then
    echo "POST-HARNESS: tree restored, suite green again ($FINAL passed). OK"
else
    echo "POST-HARNESS FAILED: $FINAL passed against $PRISTINE expected." >&2
    exit 2
fi

[ "$BROKEN" -eq 0 ] || { echo "NOTE: $BROKEN mutation(s) reported BROKEN/CRASH -- re-run in isolation." >&2; }
[ "$SURVIVED" -eq 0 ] || exit 3
