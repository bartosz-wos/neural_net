#!/usr/bin/env bash
# =============================================================================
# tools/hutchinson_mutation.sh -- mutation harness for utils/hutchinson.
#
# This repo has recorded SIXTEEN instances of a mutation harness producing a
# WRONG verdict, several of which reported one harness defect as N mutation
# verdicts. Every defence below was learned from one of those; the comments say
# which, so a future session does not have to rediscover them.
#
# THE DEFENCES, AND WHERE EACH CAME FROM:
#  1. Link set from `make print-lib-objs`, NEVER a regex over the Makefile.
#     `LIB_OBJS` is a RECURSIVE make variable (`:=` over a `$(LIB_SRCS:...)`
#     substitution reference), so its literal text contains no `.o` filenames.
#     A regex yields ZERO objects and every mutation becomes a false verdict.
#  2. Refuse to run if the mutated object is in the link set -- it is being
#     compiled separately, so every link fails with `multiple definition` and
#     every mutation reports "CAUGHT (compile error)". A LINK ERROR IS NOT A
#     MUTATION VERDICT.
#  3. `rm -f` the test object and the mutated object before every build --
#     removing the test object without rebuilding makes every link fail.
#  4. Verify each edit APPLIED by checksum, else MUT-NOOP. A sed that does not
#     match is a silent no-op, and "the suite does not cover this" is a lie.
#  5. Snapshot and restore BOTH the .cpp and the header. An earlier session
#     snapshotted only the .cpp while mutating the header, so the mutant LEAKED
#     into the tree; the next run reported N identical errors.
#  6. Parse the summary as `N passed, M failed` and GUARD against `NF="1 ==="`
#     reaching a `[ ... -gt 0 ]` test, which is a bash SYNTAX ERROR that sends
#     control to the else branch -- that inverted every result once.
#  7. Never run `make` concurrently with this harness: it races on build/*.o
#     and produces phantom MUT-BROKEN verdicts.
#  8. Rebuild and re-verify the REAL binary on exit. The harness restores the
#     source but leaves build/ holding the last mutant.
#  9. Mutations must compile WARNING-FREE. `-Wunused-variable` gives
#     MUT-BROKEN, which is a BUILD verdict and not a mutation verdict.
# 10. Spec delimiter 0x1f, not `|`: `|` collides with the `||` in every C++
#     guard, splitting the anchor mid-expression and CORRUPTING the file.
# =============================================================================
set -u

SRC=include/nn/utils/hutchinson.cpp
HDR=include/nn/utils/hutchinson.h
TEST=tests/test_hutchinson.cpp
OBJ=build/utils/hutchinson.o
TOBJ=build/test_hutchinson.o
BIN=build/test_hutchinson
DELIM=$'\x1f'

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }

[ -f "$SRC" ] && [ -f "$TEST" ] || { red "missing sources"; exit 1; }

# --- defence 7: refuse to run concurrently with a build ---------------------
if pgrep -f "make.*test_hutchinson" >/dev/null 2>&1; then
    red "ABORT: a make is running; it races on build/*.o (defence 7)."; exit 1
fi

# --- defence 1+2: the link set ---------------------------------------------
LIB_LINK=$(make -s print-lib-objs 2>/dev/null | grep -v '^$')
LIB_COUNT=$(printf '%s\n' "$LIB_LINK" | wc -l)
if [ "$LIB_COUNT" -lt 10 ]; then
    red "ABORT: link set has $LIB_COUNT objects (need >= 10). Defence 1."
    exit 1
fi
echo "link set: $LIB_COUNT objects"

# defence 2, done correctly. The Makefile links $(LIB_OBJS) wholesale, so the
# mutated translation unit is ALWAYS in that list -- which is exactly why the
# historical failure ("the mutated object was in its own link set, so every
# link failed with `multiple definition` and every mutation reported CAUGHT")
# happened when a harness compiled the mutant SEPARATELY and then also linked
# the original. We do NOT compile it separately; we rebuild it in place, so
# there is no double definition to guard against.
#
# What defence 2 actually protects is a harness that links a STALE copy. So
# assert the object we mutate is the object make rebuilds, and that we remove
# it before every build (defence 3 does that per-mutation).
if [ ! -f "$OBJ" ]; then
    red "ABORT: $OBJ absent; the harness would link a stale object. Defence 2."
    exit 1
fi
# Sanity: the pristine build must already contain it, else nothing we do is
# testing the code we think we are.
if ! make -s print-lib-objs | grep -qx "$OBJ"; then
    red "ABORT: $OBJ is not in the link set at all (defence 1/2)."
    exit 1
fi

# --- snapshot BOTH files (defence 5), restored ATOMICALLY ------------------
BAK_SRC=$(mktemp); BAK_HDR=$(mktemp)
cp "$SRC" "$BAK_SRC"; cp "$HDR" "$BAK_HDR"
restore() {
    cp "$BAK_SRC" "$SRC.tmp" && mv -f "$SRC.tmp" "$SRC"
    cp "$BAK_HDR" "$HDR.tmp" && mv -f "$HDR.tmp" "$HDR"
    rm -f "$BAK_SRC" "$BAK_HDR"
}
trap restore EXIT INT TERM

# --- pristine check: refuse to run unless the suite is green ---------------
rm -f "$TOBJ" "$OBJ"
make -s build/test_hutchinson >/dev/null 2>&1
PRISTINE=$(./$BIN 2>/dev/null | sed -n 's/.*Summary: \([0-9]*\) passed, \([0-9]*\) failed.*/\1 \2/p')
read -r PP PF <<<"$PRISTINE"
if [ "${PF:-1}" != "0" ]; then
    red "ABORT: pristine suite is not green ($PRISTINE). Refusing to score mutations."
    restore; exit 1
fi
green "pristine suite green: $PP passed, 0 failed"

# --- apply a spec: <file><DELIM><anchor><DELIM><replacement> --------------
apply_spec() {
    local file="$1" anchor="$2" repl="$3"
    python3 - "$file" "$anchor" "$repl" <<'PY'
import sys
path, anchor, repl = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
if src.count(anchor) != 1:
    print(f"ANCHOR-COUNT:{src.count(anchor)}")
    sys.exit(3)
open(path, "w").write(src.replace(anchor, repl))
PY
}

verify_changed() {  # $1 = checksum before, $2 = file
    [ "$(md5sum "$2" | cut -d' ' -f1)" != "$1" ]
}

run_one() {   # id  anchor  replacement  [file]
    local id="$1" anchor="$2" repl="$3" file="${4:-$SRC}"
    local before after out nf

    restore
    before=$(md5sum "$file" | cut -d' ' -f1)
    if ! apply_spec "$file" "$anchor" "$repl"; then
        printf '  %-6s MUT-NOOP   (anchor not unique in %s)\n' "$id" "$file"
        return
    fi
    # defence 4: the edit must have applied
    after=$(md5sum "$file" | cut -d' ' -f1)
    if [ "$before" = "$after" ]; then
        printf '  %-6s MUT-NOOP   (edit did not apply)\n' "$id"
        return
    fi

    rm -f "$TOBJ" "$OBJ"            # defence 3
    if ! make -s build/test_hutchinson >/tmp/hutch_build.log 2>&1; then
        if grep -qE 'warning:' /tmp/hutch_build.log; then
            printf '  %-6s MUT-BROKEN (warning -- a BUILD verdict, not a mutation one)\n' "$id"
        else
            printf '  %-6s MUT-BROKEN (build failed)\n' "$id"
        fi
        return
    fi
    if [ ! -x "$BIN" ]; then
        printf '  %-6s UNAVAILABLE (no binary produced)\n' "$id"   # never a verdict
        return
    fi

    out=$(./$BIN 2>/dev/null | sed -n 's/.*Summary: \([0-9]*\) passed, \([0-9]*\) failed.*/x \1 \2/p')
    if [ -z "$out" ]; then
        printf '  %-6s MUT-CRASH  (no summary line)\n' "$id"
        return
    fi
    # defence 6: parse explicitly and reject a malformed NF before comparing
    set -- $out
    local np=$2 nf=$3
    if ! printf '%s' "$nf" | grep -qE '^[0-9]+$'; then
        red "ABORT: unparsable failure count '$nf' (defence 6)"; exit 1
    fi
    if [ "$nf" -gt 0 ]; then
        printf '  %-6s CAUGHT     (%s failed)\n' "$id" "$nf"
    else
        printf '  %-6s SURVIVED   <-- investigate\n' "$id"
    fi
}

echo
echo "=== mutation run ==="

# ---- M01: drop the 1/probes normalisation ---------------------------------
run_one M01 \
  'return acc / static_cast<double>(probes);' \
  'return acc;'

# ---- M02: contract against zp instead of the perturbation ----------------
run_one M02 \
  'q += (zp(0, j) - x(0, j)) * (gp(0, j) - g0(0, j));' \
  'q += zp(0, j) * (gp(0, j) - g0(0, j));'

# ---- M03: drop the /eps in the FOM ---------------------------------------
run_one M03 \
  'hv(0, j) = (gp(0, j) - g0(0, j)) / eps;' \
  'hv(0, j) = (gp(0, j) - g0(0, j));'

# ---- M04: Rademacher probe becomes 0 -------------------------------------
run_one M04 \
  'return coin(rng) ? 1.0 : -1.0;' \
  'return 0.0;'

# ---- M05: the eps floor guard removed ------------------------------------
run_one M05 \
  '    if (eps < kMinEps)
        throw std::invalid_argument("hutch: eps below the cancellation floor");' \
  '    if (false)
        throw std::invalid_argument("hutch: eps below the cancellation floor");'

# ---- M06: drop the gradient zeroing (Dense accumulates!) -----------------
run_one M06 \
  '        for (auto& holder : model.layers)
            if (holder.get()) holder.get()->zero_grad();' \
  '        // gradients deliberately left accumulated'

# ---- M07: the parameter restore becomes a no-op ---------------------------
run_one M07 \
  '            for (size_t k = 0; k < t.rows * t.cols; ++k)
                t.data[k] = e.snapshot.data[k];' \
  '            for (size_t k = 0; k < t.rows * t.cols; ++k)
                t.data[k] = t.data[k];'

# ---- M08: the dim==0 guard removed ---------------------------------------
run_one M08 \
  '    if (dim == 0) throw std::invalid_argument("hutch: dim must be > 0");' \
  '    if (dim == 0) return 0.0;'

# ---- M09: the probes guard inverted --------------------------------------
run_one M09 \
  '    if (probes < 1) throw std::invalid_argument("hutch: probes must be >= 1");
    if (x.cols != dim)' \
  '    if (probes > 1) throw std::invalid_argument("hutch: probes must be >= 1");
    if (x.cols != dim)'

# ---- M10: FOM uses the SUM of gradients instead of the difference --------
run_one M10 \
  '        hv(0, j) = (gp(0, j) - g0(0, j)) / eps;' \
  '        hv(0, j) = (gp(0, j) + g0(0, j)) / eps;'

# ---- M11 (HEADER): the returned HVP aliases the gradient -----------------
run_one M11 \
  '    Tensor hv(g0.rows, g0.cols);
    for (size_t j = 0; j < g0.cols; ++j)
        hv(0, j) = (gp(0, j) - g0(0, j)) / eps;
    return hv;' \
  '    gp(0, 0) += (gp(0, 0) - g0(0, 0)) / eps;
    return gp;' \
  "$SRC"

# ---- M12: the Gaussian law silently reuses Rademacher --------------------
run_one M12 \
  '                          std::normal_distribution<double> nd(0.0, 1.0);
                          return nd(rng);' \
  '                          std::uniform_int_distribution<int> coin(0, 1);
                          return coin(rng) ? 1.0 : -1.0;'

# ---- defence 8: restore, rebuild, and re-verify the REAL binary ----------
restore
rm -f "$TOBJ" "$OBJ"
make -s build/test_hutchinson >/dev/null 2>&1
FINAL=$(./$BIN 2>/dev/null | sed -n 's/.*Summary: \([0-9]*\) passed, \([0-9]*\) failed.*/\1 \2/p')
read -r FP FF <<<"$FINAL"
if [ "${FF:-1}" = "0" ]; then
    green "post-harness rebuild verified: $FP passed, 0 failed"
else
    red "POST-HARNESS REBUILD FAILED: $FINAL -- the tree is not what it was"
fi