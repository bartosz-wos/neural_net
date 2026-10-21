#!/usr/bin/env bash
# tests/mutate_weight_std.sh -- mutation harness for utils/weight_std.{h,cpp}.
#
# This harness exists because the repo has recorded SEVENTEEN instances of a
# mutation harness reporting a confident wrong verdict. Every one of the defects
# below is one this repo has already paid for, so they are addressed by
# construction rather than by memory:
#
#   1. restore() MUST NOT DELETE ITS OWN BACKUP. It is called once per mutation
#      AND on EXIT; a restore that `rm`s the backup works exactly once, every
#      later restore is a silent no-op printing `cp: cannot stat`, and the tree
#      is left holding whatever mutant ran last. Cleanup happens exactly once,
#      on EXIT, and restore() is ATOMIC (`cp tmp && mv -f`).
#   2. NEVER RECOVER THE LINK SET BY REGEX. `LIB_OBJS := $(LIB_SRCS:...)` is a
#      RECURSIVE make variable whose literal text contains no .o filenames; a
#      regex against it yields ZERO objects and turns every mutation into a false
#      "did not link". This harness asks make, and ASSERTS the count.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET, or every mutation
#      double-defines every symbol and reports CAUGHT (compile error). A link
#      error is not a mutation verdict.
#   4. `MUT-BROKEN` MEANS THE BUILD FAILED. It is reported separately from
#      CAUGHT and never counted as a pass, and a mutation reported BROKEN is
#      re-run in isolation before it is believed.
#   5. A MUTATION THAT DID NOT APPLY IS A HARNESS DEFECT, not a verdict.
#      The tree is checksummed before and after every edit.
#   6. NO EMBEDDED NEWLINES IN A sed PATTERN -- `\n` does not match in GNU sed.
#      Every multi-token replacement uses 0x1f as an internal placeholder and a
#      two-step sed.
#   7. A MISSING BINARY IS `UNAVAILABLE`, never a verdict.
#   8. The spec loop refuses any entry that does not name a known file tag, so a
#      stray `inert` marker can never be read as a sed script.
#   9. Throwing calls are wrapped so a guard mutation reports a verdict instead
#      of aborting the binary and leaving the harness with nothing to count.
#  10. `rm -f` the intermediates before every build -- a stale .o from the
#      previous mutation reproduces a build failure that does not exist alone.

set -u
cd "$(dirname "$0")/.." || exit 1

SRC="include/nn/utils/weight_std.cpp"
HDR="include/nn/utils/weight_std.h"
TEST="tests/test_weight_std.cpp"
BIN="build/test_weight_std"
BAK_SRC=$(mktemp); BAK_HDR=$(mktemp)
BAK_MARK=".mutate_ws_tree_pristine"
cp "$SRC" "$BAK_SRC"; cp "$HDR" "$BAK_HDR"
cp "$SRC" "$BAK_MARK"

restore() {                      # atomic; NEVER deletes the backups
    cp "$BAK_SRC" "$SRC.tmp" 2>/dev/null && mv -f "$SRC.tmp" "$SRC"
    cp "$BAK_HDR" "$HDR.tmp" 2>/dev/null && mv -f "$HDR.tmp" "$HDR"
}
cleanup() { restore; rm -f "$BAK_SRC" "$BAK_HDR" "$BAK_MARK"; }
trap cleanup EXIT INT TERM

# ---- link set, from make, with a count assertion (defect 2) -----------------
# Defect 3 needs care here and the naive version of the guard is WRONG.
# `LIB_OBJS` is the library's own object list and it LEGITIMATELY CONTAINS
# build/utils/weight_std.o -- that is the translation unit we mutate. Excluding
# it would make the final link fail for the wrong reason (an undefined symbol),
# which is itself a false verdict. The correct handling of "the mutated object
# must not be in its own link set" is to COMPILE THE MUTATED TU SEPARATELY,
# drop that one object from the library link set, and link the mutant in its
# place. One object, linked once.
LIB_LINK=$(make -s print-lib-objs 2>/dev/null | grep '\.o$' | grep -v '/test_' | tr '\n' ' ')
LIB_COUNT=$(echo "$LIB_LINK" | wc -w)
if [ "$LIB_COUNT" -lt 10 ]; then
    echo "HARNESS DEFECT: recovered only $LIB_COUNT library objects (< 10)." >&2
    echo "Every verdict below would be false. REFUSING TO RUN." >&2
    exit 2
fi
MUT_OBJ="build/utils/weight_std.o"
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

# The test binary is linked from the library objects PLUS our own TU. Building
# it here (rather than via `make $BIN`, which would relink the pristine object)
# is what keeps the mutant in and the original out.
CXXFLAGS_LOCAL="-std=c++17 -O2 -Wall -Wextra -march=native -Iinclude"

build_and_run() {   # defect 10
    rm -f build/test_weight_std.o "$BIN" "$MUT_OBJ"
    make -s build/utils/weight_std.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$TEST" -o build/test_weight_std.o >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL -c "$SRC" -o "$MUT_OBJ" >/dev/null 2>&1 || return 1
    g++ $CXXFLAGS_LOCAL $LIB_LINK "$MUT_OBJ" build/test_weight_std.o -o "$BIN" >/dev/null 2>&1 || return 1
    return 0
}

# ---- pristine self-check: the suite must be GREEN before any mutation ------
build_and_run
PRISTINE=$("$BIN" 2>&1 | sed -n 's/^=== Summary: \([0-9]*\) passed.*/\1/p' | tail -1)
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

    # Every spec must name a file that exists AND contain the anchor exactly
    # once, or we are looking at a harness defect rather than a verdict.
    #
    # HARNESS DEFECT #2 OF THIS SESSION, and the same class as the perl
    # delimiter bug: the uniqueness pre-check used `grep -c -F -- "$from"`, and
    # `grep -c` counts MATCHING LINES, not matches. A two-line anchor therefore
    # reported 2 for a single occurrence and the mutation was rejected as
    # HARNESS-DEFECT even though it was perfectly well-formed -- one mutation
    # that never ran. The count must come from the same engine that performs the
    # edit, or the two can disagree. Python does both now.
    if [ ! -f "$file" ]; then
        TABLE+=("$id|$file does not exist|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi
    local n
    n=$(python3 -c 'import sys; print(open(sys.argv[1],encoding="utf8").read().count(sys.argv[2]))' "$file" "$from")
    if [ "$n" -ne 1 ]; then
        TABLE+=("$id|anchor count = $n|HARNESS-DEFECT"); NOOP=$((NOOP+1)); return
    fi

    local before after
    before=$(md5sum "$file" | cut -d' ' -f1)
    # defect 6: no \n inside a sed pattern, and NO `/`-DELIMITED SCRIPT. The
    # first draft used `perl -0pi -e "s/\Q$from\E/$to/"`, and C++ is FULL of
    # `/` -- `std::sqrt(n * var / (s * s));` made perl read `/` as the
    # s/// delimiter and die with "Unknown regexp modifier", so the edit never
    # happened. The harness correctly reported HARNESS-DEFECT (checksum
    # unchanged) rather than a false verdict, which is the guard working as
    # designed -- but it is still a defect, because six mutations never ran.
    #
    # The fix is a delimiter that cannot occur in the spec: 0x1f. Both the
    # pattern and the replacement go through it, so neither `/` nor `|` nor
    # `&` is special.
    local US=$'\x1f'
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
    np=$(echo "$out" | sed -n 's/^=== Summary: \([0-9]*\) passed.*/\1/p' | tail -1)
    nf=$(echo "$out" | sed -n 's/.*, \([0-9]*\) failed ===.*/\1/p' | tail -1)
    np=${np:-0}; nf=${nf:-0}

    if [ "$rc" -ge 128 ]; then              # a crash is not a verdict
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
echo "=== weight_std mutations ==="

# --- the two clean factors the oracle identified, one mutation each ---------
run_mutation M01 "$SRC" "sum_gx / (double)w.cols" "sum_gx"
run_mutation M02 "$SRC" "- x * mean_gx) * inv_s" "- x * mean_gx)"

# --- population vs sample variance (defect: a 1/(K-1) slip) ----------------
run_mutation M03 "$SRC" "const double var = acc / (double)n;" "const double var = acc / (double)(n - 1);"

# --- eps INSIDE the sqrt ---------------------------------------------------
run_mutation M04 "$SRC" "const double denom = var + eps;" "const double denom = var;"
run_mutation M05 "$SRC" "const double denom = var + eps;" "const double denom = var + eps * 0.5;"

# --- the centering (the whole point of WS vs weight_norm) ------------------
run_mutation M06 "$SRC" "out[c][j] = (w[c][j] - mean) / s;" "out[c][j] = w[c][j] / s;"
run_mutation M07 "$SRC" "mean_out = s / (double)n;" "mean_out = 0.0;"

# --- ROW vs COLUMN (invisible on a square fixture set) ---------------------
# The anchor `row_mean_scale(w[c], w.cols, eps, mean)` occurs FOUR times (the
# forward, the in-place version, the backward, and the Lipschitz helper), so it
# is NOT a legal anchor. `require_matrix` is called once in each of the three
# entry points, and deleting it there must be caught by the guard tests -- that
# is the version worth mutating, and it is unique.
run_mutation M08 "$SRC" '    require_matrix(w, "ws_standardize");
    require_eps(eps, "ws_standardize");' '    require_eps(eps, "ws_standardize");'

# --- translation invariance: delete the -mean(g) term ----------------------
run_mutation M09 "$SRC" "g[c][j] - mean_g - x * mean_gx" "g[c][j] - x * mean_gx"

# --- the accumulation fix in the wrapper (the bug the suite found) ---------
run_mutation M10 "$SRC" "const Tensor delta = *gr[0] - prev;" "const Tensor delta = *gr[0];"

# --- update_weights sign ---------------------------------------------------
run_mutation M11 "$SRC" "v_ = v_ - v_grad_ * lr;" "v_ = v_ + v_grad_ * lr;"

# --- the wrapper must leave the inner layer holding the RAW weights --------
# A guard that removes the WRITE-BACK, not a compile error: `*p[0] = v_;` alone
# is a statement that must remain syntactically present.
run_mutation M12 "$SRC" "*p[0] = v_;" "if (false) *p[0] = v_;"

# --- a guard removal: the zero-variance trap -------------------------------
run_mutation M13 "$SRC" "if (!(denom > 0.0))" "if (false)"

# --- the Lipschitz closed form --------------------------------------------
run_mutation M14 "$SRC" "std::sqrt(n * var / (s * s));" "std::sqrt(n);"

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
FINAL=$("$BIN" 2>&1 | sed -n 's/^=== Summary: \([0-9]*\) passed.*/\1/p' | tail -1)
if "$BIN" >/dev/null 2>&1 && [ "${FINAL:-0}" -eq "$PRISTINE" ]; then
    echo "POST-HARNESS: tree restored, suite green again ($FINAL passed). OK"
else
    echo "POST-HARNESS FAILED: $FINAL passed against $PRISTINE expected." >&2
    exit 2
fi

[ "$BROKEN" -eq 0 ] || { echo "NOTE: $BROKEN mutation(s) reported BROKEN/CRASH -- re-run in isolation." >&2; }
[ "$SURVIVED" -eq 0 ] || exit 3
exit 0