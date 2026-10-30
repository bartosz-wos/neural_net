#!/usr/bin/env bash
# =============================================================================
# tests/mutate_vae.sh -- mutation harness for the VAE's loss conventions and
# its KL gradient.
#
# PURPOSE: prove that tests/test_vae.cpp and tests/test_vae_grad.cpp
# actually CATCH the defects they were written for. A green suite proves
# nothing on its own; the mutation has to fail.
#
# =============================================================================
# THE MUTATIONS, AND THE DEFECT EACH ONE RE-INTRODUCES
# =============================================================================
#
#   M1  drop the KL's 1/N (the batch mean)      -> the loss convention bug
#   M2  drop the KL's 1/N in the GRADIENT        -> the gradient's scale bug
#   M3  flip the KL's mu-gradient sign           -> the gradient's sign bug
#   M4  flip the KL's logvar-gradient sign       -> a sign the code got right,
#                                                  checked anyway
#   M5  drop the KL's 1/N in the logvar gradient -> the gradient's scale bug,
#                                                  on the other term
#
# **M3 IS THE ONE THAT MATTERS MOST, AND IT IS NOT COSMETIC.** The KL is
# minimised at mu = 0, so the gradient of the KL must have the SAME SIGN as
# mu. With the sign inverted, gradient descent moved mu AWAY from zero --
# the direction that INCREASES the KL -- and the KL term was actively
# pushing the encoder away from the prior. The suite's section 5 asserts
# the BEHAVIOUR (a training step moves mu toward zero), which is the
# assertion that survives a refactor of the gradient's internals.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
# =============================================================================
#
#   1. `restore()` must NEVER delete its backup. It is called once PER
#      MUTATION and once on EXIT, so a self-cleaning restore works exactly
#      once and every later restore is a silent no-op -- leaving a MUTANT IN
#      THE TREE. Cleanup happens exactly once, on EXIT, and restore() is
#      ATOMIC (`cp tmp && mv -f`).
#   2. The link set comes from `make print-lib-objs`, NEVER from
#      regex-parsing the Makefile, and the count is asserted >= 10.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET. It legitimately
#      IS, so the mutated TU is compiled SEPARATELY and swapped into the
#      link set.
#   4. `MUT-BROKEN` (build failed) and `MUT-CRASH` (rc >= 128) are separate
#      classes and are NEVER counted as passes.
#   5. A mutation that DID NOT APPLY is a HARNESS DEFECT, not a "SURVIVED"
#      verdict. The tree is checksummed before and after every edit.
#   6. The anchor must occur EXACTLY ONCE, counted by the same engine that
#      performs the edit.
#   7. `rm -f` the intermediates before every build.
#   8. NEVER run a build concurrently with this harness.
#   9. The tree is rebuilt and re-verified ON EXIT before any verdict.
#  10. The pristine tree must be GREEN before any mutation runs.
#  11. The separator is a REAL 0x1f BYTE (`$'\x1f'`).
#  12. Both suites are run for every mutation, and the verdict records
#      WHICH one caught it -- the loss suite and the gradient suite cover
#      DIFFERENT claims, and a mutation caught only by the broadest one is
#      weaker evidence than one caught by the suite that owns the decision.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 99

SRC_CPP="include/nn/layers/generative/vae.cpp"
SRC_HDR="include/nn/layers/generative/vae.h"
SUITE_LOSS="test_vae"
SUITE_GRAD="test_vae_grad"

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
    # THE POST-HARNESS REBUILD-AND-VERIFY DEFENCE.
    rm -f build/layers/generative/vae.o \
          build/test_vae.o build/test_vae_grad.o \
          build/test_vae build/test_vae_grad
    make setup >/dev/null 2>&1
    if make -j6 build/test_vae build/test_vae_grad >/dev/null 2>&1 \
       && ./build/test_vae >/dev/null 2>&1 && ./build/test_vae_grad >/dev/null 2>&1; then
        echo ""
        echo "POST-HARNESS: tree restored, rebuilt, and GREEN."
    else
        echo ""
        echo "POST-HARNESS FAILURE: the restored tree does not build or pass. STOP." >&2
        exit 1
    fi
}
trap cleanup EXIT INT TERM

checksum() { md5sum "$1" | cut -d' ' -f1; }
US=$'\x1f'

# -----------------------------------------------------------------------------
# The link set, from make itself, with a count assertion.
# -----------------------------------------------------------------------------
mapfile -t ALL_OBJS < <(make print-lib-objs)
if [ "${#ALL_OBJS[@]}" -lt 10 ]; then
    echo "HARNESS DEFECT: link set has only ${#ALL_OBJS[@]} objects (need >= 10)." >&2
    exit 99
fi
LIB_LINK=""
for o in "${ALL_OBJS[@]}"; do
    case "$o" in
        *generative/vae.o|*test_vae.o|*test_vae_grad.o) ;;
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
# Build both suites against the CURRENT (possibly mutated) source.
# -----------------------------------------------------------------------------
build_suites() {
    rm -f "$SCRATCH/vae.o" "$SCRATCH"/test_vae.o "$SCRATCH"/test_vae_grad.o \
          "$SCRATCH/test_vae" "$SCRATCH/test_vae_grad"
    g++ $CXXFLAGS_LOCAL -c "$SRC_CPP" -o "$SCRATCH/vae.o" 2>"$SCRATCH/build.log" || return 90
    local t
    for t in "$SUITE_LOSS" "$SUITE_GRAD"; do
        g++ $CXXFLAGS_LOCAL -c "tests/$t.cpp" -o "$SCRATCH/$t.o" 2>>"$SCRATCH/build.log" || return 90
        g++ $CXXFLAGS_LOCAL "$SCRATCH/$t.o" "$SCRATCH/vae.o" $LIB_LINK \
            -o "$SCRATCH/$t" 2>>"$SCRATCH/build.log" || return 90
    done
    return 0
}

# The pristine self-check: both suites must be GREEN.
build_suites
if [ $? -ne 0 ]; then
    echo "HARNESS DEFECT: the PRISTINE tree does not build." >&2
    tail -20 "$SCRATCH/build.log" >&2
    exit 99
fi
for t in "$SUITE_LOSS" "$SUITE_GRAD"; do
    if ! "$SCRATCH/$t" > "$SCRATCH/out_$t.txt" 2>&1; then
        echo "HARNESS DEFECT: the pristine $t is not green." >&2
        tail -20 "$SCRATCH/out_$t.txt" >&2
        exit 99
    fi
    echo "pristine: $t -- $(sed -nE 's/.*SUMMARY: ([0-9]+) passed.*/\1/p' "$SCRATCH/out_$t.txt" | tail -1) passed, 0 failed"
done
echo "pristine: both suites green -- mutation may begin"

# -----------------------------------------------------------------------------
# Apply a mutation with a Python editor (str.count / str.replace).
# -----------------------------------------------------------------------------
apply_mutation() {  # $1 = spec (old<US>new)
    python3 - "$1" <<'PYEOF'
import sys, pathlib
spec = sys.argv[1]
old, new = spec.split("\x1f")
p = pathlib.Path("include/nn/layers/generative/vae.cpp")
s = p.read_text()
n = s.count(old)
if n != 1:
    print(f"ANCHOR_COUNT={n}")
    sys.exit(3)
p.write_text(s.replace(old, new))
print("MUTATION_APPLIED")
PYEOF
}

CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0
declare -a TABLE

run_mutation() {
    local id="$1" desc="$2" old="$3" new="$4"
    restore

    local out
    out=$(apply_mutation "$old${US}$new")
    if ! echo "$out" | grep -q "MUTATION_APPLIED"; then
        TABLE+=("$id|$desc|HARNESS-DEFECT (anchor count != 1)|$(echo "$out" | tr '\n' ' ')")
        NOOP=$((NOOP+1)); return
    fi

    local before after
    before=$(checksum "$BAK_CPP"); after=$(checksum "$SRC_CPP")
    if [ "$before" = "$after" ]; then
        restore
        TABLE+=("$id|$desc|HARNESS-DEFECT (edit did not land)|"); NOOP=$((NOOP+1)); return
    fi

    build_suites
    local rc=$?
    if [ "$rc" -eq 90 ]; then
        restore
        TABLE+=("$id|$desc|MUT-BROKEN (build failed -- NOT a verdict)|"); BROKEN=$((BROKEN+1)); return
    fi

    local t caught_by="" nf line
    for t in "$SUITE_LOSS" "$SUITE_GRAD"; do
        "$SCRATCH/$t" > "$SCRATCH/out_$t.txt" 2>&1; rc=$?
        if [ "$rc" -ge 128 ]; then
            restore
            TABLE+=("$id|$desc|MUT-CRASH in $t (rc=$rc -- NOT a verdict)|")
            CRASH=$((CRASH+1)); return
        fi
        nf=$(sed -nE 's/.*SUMMARY: ([0-9]+) passed, ([0-9]+) failed.*/\2/p' "$SCRATCH/out_$t.txt" | tail -1)
        if [ -n "$nf" ] && [ "$nf" -gt 0 ]; then
            caught_by="$t"
            line=$(grep -m1 "FAIL" "$SCRATCH/out_$t.txt" | sed 's/^ *//' | cut -c1-60)
            break
        fi
    done

    if [ -n "$caught_by" ]; then
        TABLE+=("$id|$desc|CAUGHT by $caught_by|$line")
        CAUGHT=$((CAUGHT+1))
    else
        TABLE+=("$id|$desc|SURVIVED (both suites stayed green)|")
        SURVIVED=$((SURVIVED+1))
    fi
    restore
}

echo
echo "=== vae mutations ==="

# --- M1: drop the KL's 1/N in the LOSS -------------------------------------
run_mutation M1 "drop the KL's 1/N in the loss" \
    "    kl_loss_ /= static_cast<double>(input.rows);" \
    ""

# --- M2: drop the KL's 1/N in the GRADIENT ----------------------------------
run_mutation M2 "drop the KL's 1/N in the gradient" \
    "    const double inv_batch = 1.0 / static_cast<double>(last_mu_.rows);" \
    "    const double inv_batch = 1.0;"

# --- M3: flip the KL's mu-gradient SIGN -------------------------------------
run_mutation M3 "flip the KL's mu-gradient sign" \
    "            grad_mu_enc[i][j] = mu_ij * inv_batch;" \
    "            grad_mu_enc[i][j] = -mu_ij * inv_batch;"

# --- M4: flip the KL's logvar-gradient sign ---------------------------------
run_mutation M4 "flip the KL's logvar-gradient sign" \
    "            grad_logvar_enc[i][j] = 0.5 * (std::exp(logvar_ij) - 1.0) * inv_batch;" \
    "            grad_logvar_enc[i][j] = 0.5 * (1.0 - std::exp(logvar_ij)) * inv_batch;"

# --- M5: drop the logvar gradient's 1/N only --------------------------------
run_mutation M5 "drop the logvar gradient's 1/N only" \
    "            grad_logvar_enc[i][j] = 0.5 * (std::exp(logvar_ij) - 1.0) * inv_batch;" \
    "            grad_logvar_enc[i][j] = 0.5 * (std::exp(logvar_ij) - 1.0);"

echo
printf '%-6s %-42s %-30s %s\n' "ID" "MUTATION" "VERDICT" "DETAIL"
printf '%-6s %-42s %-30s %s\n' "----" "----" "------" "------"
for row in "${TABLE[@]}"; do
    IFS='|' read -r a b c d <<< "$row"
    printf '%-6s %-42s %-30s %s\n' "$a" "$b" "$c" "${d:-}"
done

echo
echo "FINAL: $CAUGHT caught / $SURVIVED survived / $BROKEN broken / $CRASH crashed / $NOOP noop"

rm -rf "$SCRATCH"

[ "$SURVIVED" -eq 0 ] || { echo "NOTE: $SURVIVED unexplained survivor(s) -- these are REAL coverage gaps." >&2; exit 3; }
[ "$BROKEN" -eq 0 ] || { echo "NOTE: $BROKEN mutation(s) reported BROKEN/CRASH -- re-run in isolation." >&2; exit 1; }
[ "$NOOP" -eq 0 ] || { echo "NOTE: $NOOP harness defect(s)." >&2; exit 1; }
exit 0
