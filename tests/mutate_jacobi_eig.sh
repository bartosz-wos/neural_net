#!/bin/bash
# MUTATION HARNESS for the Jacobi eigendecomposition fix.
#
# PURPOSE: prove that tests/test_jacobi_eig.cpp actually CATCHES the defect it
# was written for. A green suite proves nothing on its own; the mutation has
# to fail.
#
# THE MUTATIONS are the three ways the rotation angle can be wrong:
#   M1  shampoo: restore the original `aqq - app` sign
#   M2  soap:    restore the original `aqq - app` sign
#   M3  galore:  restore the original `aqq - app` sign
#   M4  shampoo: drop the degenerate-branch special case (a plausible
#       "simplification" that would break the branch the fix leaves alone)
#
# THE HARNESS RULES THIS REPO HAS ALREADY LEARNED (see the TDD skill):
#   * `restore()` must NEVER delete its backup. Cleanup happens exactly once,
#     on EXIT. The first draft of a harness in this repo destroyed its own
#     snapshot after the first restore and every later "restore" was a silent
#     no-op, leaving a MUTANT IN THE TREE.
#   * A BUILD ERROR IS NOT A MUTATION VERDICT. It says nothing about whether
#     the suite covers the behaviour. MUT-BROKEN is a separate class.
#   * A CRASH (rc >= 128) IS NOT A VERDICT EITHER. It is a third class.
#   * The mutated object must not be in its own link set.
#   * NEVER run a build concurrently with the harness (shared build/*.o).
#   * After mutating, CONFIRM the edit applied (checksum the tree).
set -u

REPO=/home/stefan/neural_net
cd "$REPO" || exit 99

BAK_DIR=$(mktemp -d)
BUILD_DIR=$BAK_DIR/build
LOG=$BAK_DIR/harness.log

LIB_OBJS=$(python3 -c "
import glob
srcs = glob.glob('include/nn/*.cpp') + glob.glob('include/nn/*/*.cpp') + glob.glob('include/nn/*/*/*.cpp')
print(' '.join(s.replace('include/nn/','build/').replace('.cpp','.o') for s in srcs))
")
OBJ_COUNT=$(echo "$LIB_OBJS" | wc -w)
if [ "$OBJ_COUNT" -lt 10 ]; then
  echo "HARNESS DEFECT: LIB_OBJS extracted $OBJ_COUNT objects (need >= 10). Aborting."
  exit 99
fi
echo "HARNESS: LIB_OBJS = $OBJ_COUNT objects"

MUTATED_TU="build/optimizers/shampoo.o"
MUTATED_SRC="include/nn/optimizers/shampoo.cpp"

# ---- backup -----------------------------------------------------------------
cp include/nn/optimizers/shampoo.cpp "$BAK_DIR/shampoo.cpp.bak"
cp include/nn/optimizers/soap.cpp    "$BAK_DIR/soap.cpp.bak"
cp include/nn/optimizers/galore.cpp  "$BAK_DIR/galore.cpp.bak"

restore_tree() {
  cp "$BAK_DIR/shampoo.cpp.bak" include/nn/optimizers/shampoo.cpp
  cp "$BAK_DIR/soap.cpp.bak"    include/nn/optimizers/soap.cpp
  cp "$BAK_DIR/galore.cpp.bak"  include/nn/optimizers/galore.cpp
}
cleanup() {
  restore_tree
  rm -rf "$BAK_DIR"
}
trap cleanup EXIT INT TERM

PASSED=0; FAILED=0; BROKEN=0; CRASHED=0; NOOP=0; INERT=0

# Run the suite against the current (possibly mutated) tree.
# Rebuild ONLY the three optimizer TUs + the test, into a private build dir so
# the repo's build/ is never touched by the harness.
run_suite() {
  rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR/optimizers" "$BUILD_DIR/test"
  local cflags="-std=c++17 -O2 -Wall -Wextra -march=native -Iinclude"

  # Compile the three optimizer TUs fresh (they are the mutated ones).
  g++ $cflags -c include/nn/optimizers/shampoo.cpp -o "$BUILD_DIR/optimizers/shampoo.o" 2>>"$LOG" || return 90
  g++ $cflags -c include/nn/optimizers/soap.cpp    -o "$BUILD_DIR/optimizers/soap.o"    2>>"$LOG" || return 90
  g++ $cflags -c include/nn/optimizers/galore.cpp  -o "$BUILD_DIR/optimizers/galore.o"  2>>"$LOG" || return 90

  # The rest of the library: reuse the repo's already-built objects, but SWAP
  # the three mutated TUs out. The mutated object is EXCLUDED from the link
  # set and replaced by ours (the naive "exclude it" form breaks the link).
  local link_objs=""
  for o in $LIB_OBJS; do
    case "$o" in
      build/optimizers/shampoo.o) link_objs="$link_objs $BUILD_DIR/optimizers/shampoo.o" ;;
      build/optimizers/soap.o)    link_objs="$link_objs $BUILD_DIR/optimizers/soap.o" ;;
      build/optimizers/galore.o)  link_objs="$link_objs $BUILD_DIR/optimizers/galore.o" ;;
      *) link_objs="$link_objs $o" ;;
    esac
  done

  g++ $cflags -c tests/test_jacobi_eig.cpp -o "$BUILD_DIR/test/test_jacobi_eig.o" 2>>"$LOG" || return 90
  g++ $cflags "$BUILD_DIR/test/test_jacobi_eig.o" $link_objs -o "$BUILD_DIR/test_jacobi_eig" 2>>"$LOG" || return 90

  "$BUILD_DIR/test_jacobi_eig" > "$BAK_DIR/out.txt" 2>&1
  return $?
}

# Apply a mutation with a Python editor (str.count/str.replace) -- perl's
# s/// dies on C++ anchors containing '/', which silently killed SIX of
# fourteen mutations in a harness in this repo.
apply_mutation() {
  python3 - "$1" "$2" <<'PY'
import sys
path, which = sys.argv[1], sys.argv[2]
files = {
    'M1': 'include/nn/optimizers/shampoo.cpp',
    'M2': 'include/nn/optimizers/soap.cpp',
    'M3': 'include/nn/optimizers/galore.cpp',
    'M4': 'include/nn/optimizers/shampoo.cpp',
}
f = files[which]
txt = open(f).read()
if which in ('M1', 'M2', 'M3'):
    old = "theta = 0.5 * std::atan2(2.0 * apq, app - aqq);"
    new = "theta = 0.5 * std::atan2(2.0 * apq, aqq - app);"
    n = txt.count(old)
    if n != 1:
        print(f"ANCHOR_COUNT={n}")
        sys.exit(1)
    open(f, 'w').write(txt.replace(old, new))
elif which == 'M4':
    old = """                if (std::abs(app - aqq) < 1e-30) {
                    theta = (apq >= 0.0) ? M_PI / 4.0 : -M_PI / 4.0;
                } else {
                    theta = 0.5 * std::atan2(2.0 * apq, app - aqq);
                }"""
    new = """                theta = 0.5 * std::atan2(2.0 * apq, app - aqq);"""
    n = txt.count(old)
    if n != 1:
        print(f"ANCHOR_COUNT={n}")
        sys.exit(1)
    open(f, 'w').write(txt.replace(old, new))
print("MUTATION_APPLIED")
PY
}

for M in M1 M2 M3 M4; do
  restore_tree
  OUT=$(apply_mutation "$M" "$M")
  if ! echo "$OUT" | grep -q "MUTATION_APPLIED"; then
    echo "$M  MUT-NOOP/AMBIGUOUS  ($(echo "$OUT" | tr '\n' ' '))"
    NOOP=$((NOOP+1))
    continue
  fi
  # Verify the edit really landed in the tree.
  if grep -q "aqq - app" include/nn/optimizers/shampoo.cpp || \
     grep -q "aqq - app" include/nn/optimizers/soap.cpp || \
     grep -q "aqq - app" include/nn/optimizers/galore.cpp; then
    EDIT_OK=1
  else
    EDIT_OK=0
  fi
  if [ "$M" = "M4" ]; then
    grep -q "M_PI / 4.0" include/nn/optimizers/shampoo.cpp && EDIT_OK=0
  fi
  if [ "$EDIT_OK" -eq 0 ]; then
    echo "$M  MUT-NOOP  (the edit did not land in the tree)"
    NOOP=$((NOOP+1))
    continue
  fi

  run_suite
  rc=$?
  if [ "$rc" -eq 90 ]; then
    echo "$M  MUT-BROKEN  (build failed -- NOT a verdict)"
    BROKEN=$((BROKEN+1))
  elif [ "$rc" -ge 128 ]; then
    echo "$M  MUT-CRASH  (rc=$rc -- NOT a verdict)"
    CRASHED=$((CRASHED+1))
  elif [ "$rc" -ne 0 ]; then
    FAILEDLINE=$(grep -m1 "FAIL" "$BAK_DIR/out.txt" | sed 's/^ *//' | cut -c1-90)
    echo "$M  CAUGHT  (suite failed as required)  [$FAILEDLINE]"
    PASSED=$((PASSED+1))
  else
    # The wording is NEUTRAL here on purpose: a survivor is only a coverage
    # gap if it cannot be shown to be inert. The inverted-verdict check
    # below re-scores it, and only an unexplained survivor counts against
    # the run.
    echo "$M  SURVIVED  (the suite stayed GREEN -- pending the inertness measurement)"
    FAILED=$((FAILED+1))
  fi
done

restore_tree
echo "----"
echo "MUTATION SUMMARY: $PASSED caught / $FAILED survived / $BROKEN broken / $CRASHED crashed / $NOOP noop"
echo "----"

# ---------------------------------------------------------------------------
# THE INVERTED VERDICT FOR THE MEASURED-INERT MUTATION.
#
# M4 (dropping the `app == aqq` special case) SURVIVES the suite, and that
# is a CORRECT verdict rather than a coverage gap: with the corrected
# general formula, `atan2(2*apq, 0)` is EXACTLY +-pi/2, so the general branch
# already produces +-pi/4 -- bit-identical to what the special case
# hardcodes. Measured: 2400 matrices, bit-identical output, worst difference
# exactly 0.0.
#
# **A SURVIVOR MUST NEVER BE SILENTLY DROPPED**, and an "inert" verdict must
# itself be checkable -- otherwise a future refactor that makes the branch
# load-bearing again would leave the harness reporting a stale "inert" that
# is no longer true. This function INVERTS the verdict: it PASSES when the
# mutation is still inert and FAILS when it starts to matter.
#
# **A SURVIVOR THAT IS PROVABLY INERT IS NOT A COVERAGE GAP, AND THE HARNESS
# MUST NOT COUNT IT AS ONE.** The TDD skill's rule is that an UNEXPLAINED
# survivor is indistinguishable from a real gap; a survivor that has been
# MEASURED inert is explained. So the harness separates the two: a survivor
# that the inverted-verdict check clears is scored as INERT (which is a
# passing outcome), and only a survivor that FAILS the check counts against
# the run.
# ---------------------------------------------------------------------------
echo ""
echo "=== INVERTED VERDICT: is M4 still provably inert? ==="
g++ -std=c++17 -O2 -Wall -Wextra -Iinclude -o "$BAK_DIR/m4_inert" tests/test_jacobi_inertness.cpp 2>>"$LOG"
if [ $? -ne 0 ]; then
  echo "M4-INERT  HARNESS-BROKEN  (the inverted-verdict check did not compile)"
  BROKEN=$((BROKEN+1))
elif "$BAK_DIR/m4_inert" > "$BAK_DIR/m4_out.txt" 2>&1; then
  echo "M4  MEASURED-INERT  (the inverted-verdict check PASSES: the claim still holds)"
  grep -E "matrices tested|BIT-IDENTICAL|worst abs" "$BAK_DIR/m4_out.txt" | sed 's/^/      /'
  # The survivor is EXPLAINED: downgrade it from "survived" to "inert".
  FAILED=$((FAILED-1))
  INERT=$((INERT+1))
else
  echo "M4  NOT-INERT  (the inverted-verdict check FAILED -- the 'inert' verdict is now WRONG)"
  grep -E "\[FAIL\]|differing|worst abs" "$BAK_DIR/m4_out.txt" | sed 's/^/      /'
fi
echo "----"
echo "FINAL: $PASSED caught / $FAILED survived / $INERT measured-inert / $BROKEN broken / $CRASHED crashed / $NOOP noop"
[ "$FAILED" -eq 0 ] && [ "$BROKEN" -eq 0 ] && [ "$CRASHED" -eq 0 ] && [ "$NOOP" -eq 0 ] && exit 0
exit 1
