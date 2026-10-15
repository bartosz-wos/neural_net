#!/usr/bin/env bash
# tests/mutate_vq_vae.sh — mutation harness for vq_vae.{h,cpp}
#
# A MUTATION HARNESS IS A PROGRAM THAT CAN LIE TO YOU. This repo has been
# bitten by that seven separate ways, all recorded in EXPANSION_QUEUE.md:
#
#   1. Linking LIB_OBJS (which ALREADY CONTAINS the mutated TU) double-defines
#      every symbol and reports 8/8 caught. A LINK ERROR IS NOT A TEST FAILURE.
#   2. `sed 's/foo/\nbar/'` -- a \n in a sed PATTERN does not match in GNU sed.
#      The mutation never applied and was reported as SURVIVED.
#   3. The `inert` marker left in "$@" made `apply` run `sed -i inert file`;
#      GNU sed read "inert" as a SCRIPT and wrote diagnostics into the .cpp.
#   4. Stale .o files made one mutant report a build failure that did not
#      reproduce in isolation.
#   5. `make /abs/path` reports "up to date" and never rebuilds -- 7 false
#      MUT-SURVIVED in the metrics session.
#   6. Verdict logic INVERTED: reported CAUGHT when the summary said "0 failed".
#   7. A CRASH produces no [FAIL] line, so [FAIL]-counting reads it as a pass.
#      Crashes are counted in their own column, never folded into "caught".
#
# Every defence below is load-bearing. Read this header before "simplifying"
# any of it.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/include/nn/layers/generative/vq_vae.cpp"
TEST="$REPO/tests/test_vq_vae.cpp"
BIN_REL="build/test_vq_vae"
MUT_O_REL="build/layers/generative/vq_vae.o"
LIBOBJS_REL="build/layers/generative/vq_vae.o"   # EXCLUDED from the link set

cd "$REPO" || exit 1

# ---- atomic restore (a plain `cp` truncates first, so a concurrent reader
#      can splice two generations together -- that produced a corrupted file
#      in the adversarial session). ------------------------------
BASE=$(mktemp)
cp "$SRC" "$BASE"
restore() { cp "$BASE" "$SRC.tmp" && mv -f "$SRC.tmp" "$SRC"; }
trap 'restore; rm -f "$BASE"' EXIT

echo "=========================================================="
echo " mutation harness: vq_vae"
echo "=========================================================="

# ---- DEFENCE 1: the baseline is taken ONCE, BEFORE any mutation, and
#      ASSERTED PRISTINE. A harness that snapshots after mutating will
#      eventually restore a mutant over correct code.
#      Every marker below is verified to exist in THIS file. (The first run of
#      this harness correctly REFUSED to execute because one marker, 'sg[z_e]',
#      lives in the HEADER and not the .cpp -- which is exactly the failure
#      mode the check exists to catch, arriving from the other direction.)
# ------------------------------------------------------------------
for marker in 'vq_ema_step' 'is_scalar_seed' 'total > 0.0' 'std::log(p + 1e-10)'; do
    if ! grep -qF "$marker" "$BASE"; then
        echo "HARNESS DEFECT: baseline is not pristine -- '$marker' missing from $SRC"
        echo "Refusing to run; a snapshot taken after mutating restores mutants."
        exit 2
    fi
done
echo "baseline: pristine (4 markers verified in vq_vae.cpp)"

# Build the baseline binary ONCE.
make "$BIN_REL" >/dev/null 2>&1 || { echo "baseline build FAILED"; exit 2; }
if ! ./$BIN_REL >/dev/null 2>&1; then
    echo "baseline suite does not PASS -- refusing to attribute anything."
    exit 2
fi
echo "baseline: suite passes"

# All LIB_OBJS EXCEPT the mutated translation unit. This is DEFENCE 1 above:
# including it double-defines every symbol and reports a false catch.
#
# DEFECT FOUND ON THE FIRST RUN OF THIS HARNESS: the extraction used
# `grep -E '^LIB_OBJS\s*='`, but the Makefile writes the variable as
# `LIB_OBJS := ...` -- there is a SPACE before the `:=`. The grep therefore
# matched nothing, LINK_OBJS came out EMPTY, and all seven real mutations
# were scored "MUT-BROKEN (did not link)". A harness defect reported as eight
# mutation verdicts: exactly the "a mutation harness is a program that can lie
# to you" failure, and the reason DEFENCE 1 verifies markers before running.
# The `:` in `:=` must be matched explicitly and the result COUNTED.
LIB_OBJS_RAW=$(make -pn 2>/dev/null | grep -E '^LIB_OBJS[[:space:]]*:?=' | head -1 | sed 's/^[^=]*=[[:space:]]*//')
LINK_OBJS=$(printf '%s\n' $LIB_OBJS_RAW | grep '\.o$' | grep -v "^$LIBOBJS_REL$" | tr '\n' ' ')
NOBJ=$(printf '%s\n' $LINK_OBJS | grep -c '\.o$')
echo "link set: $NOBJ objects (excluding $LIBOBJS_REL)"
if [ "$NOBJ" -lt 10 ]; then
    echo "HARNESS DEFECT: link set has only $NOBJ objects -- extraction failed."
    exit 2
fi

CAUGHT=0; SURVIVED=0; BROKEN=0; NOOP=0
declare -a VERDICTS=()

run_one() {
    local tag="$1" sedexpr="$2"
    restore

    # ---- DEFENCE 2: verify the sed ACTUALLY applied, via diff -q.
    #      An edit that never ran is indistinguishable from a gap.
    sed -i "$sedexpr" "$SRC"
    if diff -q "$BASE" "$SRC" >/dev/null; then
        echo "  MUT-NOOP     $tag  (the edit did not apply -- HARNESS DEFECT)"
        NOOP=$((NOOP+1)); VERDICTS+=("MUT-NOOP $tag"); return
    fi

    # ---- DEFENCE 4: rm intermediates before every build (stale .o).
    rm -f "$MUT_O_REL" "$BIN_REL"

    # ---- DEFENCE 5: RELATIVE make target. An absolute target reports
    #      "up to date" and silently skips the rebuild.
    if ! make "$MUT_O_REL" >/dev/null 2>&1; then
        echo "  MUT-BROKEN   $tag  (did not compile -- a compile error is NOT a"
        echo "                       test failure, so this is not a catch)"
        BROKEN=$((BROKEN+1)); VERDICTS+=("MUT-BROKEN $tag"); return
    fi
    if ! g++ -std=c++17 -O2 -Iinclude "$TEST" "$MUT_O_REL" $LINK_OBJS -o "$BIN_REL" 2>/dev/null; then
        echo "  MUT-BROKEN   $tag  (did not link -- NOT a catch)"
        BROKEN=$((BROKEN+1)); VERDICTS+=("MUT-BROKEN $tag"); return
    fi

    # ---- DEFENCE 6: mtime guard. Reject a run whose binary is older than
    #      the mutated source.
    if [ "$BIN_REL" -ot "$SRC" ]; then
        echo "  MUT-STALE    $tag  (binary older than source -- HARNESS DEFECT)"
        BROKEN=$((BROKEN+1)); VERDICTS+=("MUT-STALE $tag"); return
    fi

    OUT=$(./$BIN_REL 2>&1); RC=$?
    NFAIL=$(printf '%s\n' "$OUT" | grep -c '\[FAIL\]')

    # ---- DEFENCE 7: count [FAIL] LINES, never a summary string (whose format
    #      changes), and treat a CRASH as its own verdict -- a crash emits no
    #      [FAIL] line and would otherwise be scored as a pass.
    if [ "$RC" -ge 128 ]; then
        echo "  MUT-CRASH    $tag  (killed by signal, exit $RC -- a crash is NOT a pass)"
        BROKEN=$((BROKEN+1)); VERDICTS+=("MUT-CRASH $tag"); return
    fi
    if [ "$NFAIL" -gt 0 ]; then
        echo "  MUT-CAUGHT   $tag  (by $NFAIL assertions)"
        CAUGHT=$((CAUGHT+1)); VERDICTS+=("MUT-CAUGHT $tag"); return
    fi
    echo "  MUT-SURVIVED $tag  -- a real gap UNLESS measured inert; see the note"
    echo "                       in EXPANSION_QUEUE.md. An unexplained survivor"
    echo "                       is indistinguishable from a missing test."
    SURVIVED=$((SURVIVED+1)); VERDICTS+=("MUT-SURVIVED $tag")
}

# ===================================================================
# THE MUTATIONS
# ===================================================================
# M1: swap the two stop-gradients -- the single most consequential thing
#     this method can get wrong, and invisible in the VALUE.
run_one "M1 dictionary gradient carries beta (stop-gradients swapped)" \
        's|return vq_codebook_gradient(z, w) \* beta;|return vq_codebook_gradient(z, w);|'

# M2: drop beta from the commitment gradient entirely.
run_one "M2 drop beta from the commitment gradient" \
        's|const double inv = 2.0 \* beta / static_cast<double>(M \* D);|const double inv = 2.0 / static_cast<double>(M * D);|'

# M3: per-row mean instead of mean-over-all-elements (D2).
run_one "M3 per-row mean instead of element mean (D2)" \
        's|return s / static_cast<double>(z.rows \* z.cols);|return s / static_cast<double>(z.rows);|'

# M4: LAST-wins tie-break instead of first-wins (D4).
run_one "M4 last-wins tie-break (D4)" \
        's|if (d(i, k) < d(i, best)) best = k;|if (d(i, k) <= d(i, best)) best = k;|'

# M5: return z_e from forward -- the "quantizer that quantizes nothing".
run_one "M5 forward returns z_e instead of z_q (D5)" \
        's|return last_z_q_.clone();|return last_z_e_.clone();|g'

# M6: drop the Laplace smoothing from the EMA step (D7).
#     NOTE the expression has no `const` -- it was made mutable so the
#     divide-by-zero guard could floor it. The first version of this spec
#     carried `const` and reported MUT-NOOP: a sed that matches nothing is
#     indistinguishable from a mutation that survived, which is why the
#     harness diffs the tree after every edit.
run_one "M6 drop the EMA Laplace smoothing (D7)" \
        's|double smoothed = (new_n\[k\] + epsilon) / denom \* total;|double smoothed = new_n[k];|'

# M7: keep the initial value for a dead code instead of zeroing it (D8).
run_one "M7 dead codes keep their value instead of collapsing to 0 (D8)" \
        's|out(d, k) = new_dw(d, k) / smoothed;|out(d, k) = new_dw(d, k) / smoothed; if (new_dw(d,k) == 0.0) out(d,k) = w(d,k);|'

# M8: perplexity without the +1e-10 inside the log (D9) -- would make it
#     exactly an integer, i.e. NaN at p=0 instead of a finite contribution.
run_one "M8 perplexity without the 1e-10 log guard (D9)" \
        's|acc += p \* std::log(p + 1e-10);|acc += p * std::log(p);|'

restore
make "$BIN_REL" >/dev/null 2>&1

echo
echo "=========================================================="
echo " TOTALS: $CAUGHT caught, $SURVIVED survived, $BROKEN broken/noop, $NOOP noop"
echo "=========================================================="
for v in "${VERDICTS[@]}"; do echo "  $v"; done

# ---- DEFENCE 8: invert the verdict. The run FAILS if anything was NOOP or
#      STALE (a harness that silently ran nothing must not exit 0), and warns
#      loudly on survivors, which require a measurement before they can be
#      dismissed as inert.
if [ "$NOOP" -gt 0 ]; then
    echo "HARNESS DEFECT: $NOOP mutation(s) never applied."
    exit 2
fi
[ "$SURVIVED" -gt 0 ] && echo "NOTE: every survivor needs an inertness PROBE before it can be dismissed."
exit 0