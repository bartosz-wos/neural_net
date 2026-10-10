#!/usr/bin/env bash
# Mutation test for Grad-CAM. Each mutation breaks the implementation in a
# specific, plausible way; a correct test suite must FAIL under every one.
# A mutation that passes = a vacuous test, reported as such.
#
# LESSON FROM THE FIRST RUN OF THIS SCRIPT (kept, because it nearly caused a
# corrupted commit): the baseline backup must be taken and VERIFIED ONCE, up
# front, from a source file known to be clean. The first version snapshotted
# /tmp/gradcam_orig.cpp at the top and restored from it after every mutation --
# but the M3 mutation had already been written to the source BEFORE that
# snapshot was verified, so the "original" backup contained the mutant and every
# later restore silently wrote the BROKEN implementation back over the correct
# one. The tests then kept passing (M3's mutant had been caught, so the suite
# disagreed with itself) and the damage was only found by reading the file.
#
# Two guards follow from that:
#   1. `assert_pristine` greps for a line that only exists in the correct
#      implementation, and aborts if it is missing.
#   2. Every mutation is checked with `diff -q` against the pristine baseline
#      BEFORE the build, so a sed that silently matches nothing is reported as
#      "DID NOT APPLY" instead of counting as a pass or a vacuous test.
set -u
cd "$(dirname "$0")/.." || exit 1

SRC=include/nn/interpretability/gradcam.cpp
BASE=/tmp/gradcam_pristine.cpp

# Snapshot ONCE, and prove it is the correct implementation before touching it.
cp "$SRC" "$BASE"
if ! grep -q 'relu ? std::max(0.0, acc) : acc' "$BASE"; then
  echo "FATAL: baseline is not the pristine implementation (trailing ReLU missing)."
  echo "       A previous mutation run left the source mutated. Restore it from"
  echo "       git (this file is new/untracked: re-apply the one-line fix) and"
  echo "       re-run. Refusing to mutate a dirty baseline."
  exit 2
fi

# Fail loudly if the working tree already differs from git HEAD for this file.
if ! git diff --quiet -- "$SRC" 2>/dev/null && git ls-files --error-unmatch "$SRC" >/dev/null 2>&1; then
  echo "NOTE: $SRC differs from git HEAD (expected for a new file this run)."
fi

CAUGHT=0; MISSED=0; N=0

restore() { cp "$BASE" "$SRC"; }

echo "=== Grad-CAM mutation testing (baseline verified pristine) ==="

# Mutations that MUST break the test suite.
run_mutation() {
  local name="$1"; shift
  local sedexpr="$1"; shift
  N=$((N+1))
  restore
  sed -i "$sedexpr" "$SRC"
  if diff -q "$BASE" "$SRC" >/dev/null; then
    echo "[MUT-$N] $name -> MUTATION DID NOT APPLY (sed matched nothing)"
    MISSED=$((MISSED+1)); restore; return
  fi
  if ! g++ -std=c++17 -Iinclude tests/test_gradcam.cpp $(find include/nn -name '*.cpp' | tr '\n' ' ') -o /tmp/tg_mut 2>/dev/null; then
    echo "[MUT-$N] $name -> CAUGHT (compile error)"
    CAUGHT=$((CAUGHT+1)); restore; return
  fi
  out=$(stdbuf -o0 -e0 /tmp/tg_mut 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    n_fail=$(echo "$out" | grep -c "\[FAIL\]")
    first=$(echo "$out" | grep -m1 "\[FAIL\]" | sed 's/^ *//')
    echo "[MUT-$N] $name -> CAUGHT ($n_fail failing checks)"
    echo "         first: $first"
    CAUGHT=$((CAUGHT+1))
  else
    echo "[MUT-$N] $name -> !!! PASSED (VACUOUS TEST) !!!"
    MISSED=$((MISSED+1))
  fi
  restore
}

# A mutation PROVEN not to change any output. Passing here is the CORRECT
# result; failing here means the proof was wrong and must be investigated.
run_mutation_expect_inert() {
  local name="$1"; shift
  local sedexpr="$1"; shift
  N=$((N+1))
  restore
  sed -i "$sedexpr" "$SRC"
  if diff -q "$BASE" "$SRC" >/dev/null; then
    echo "[MUT-$N] $name -> MUTATION DID NOT APPLY (sed matched nothing)"
    MISSED=$((MISSED+1)); restore; return
  fi
  if ! g++ -std=c++17 -Iinclude tests/test_gradcam.cpp $(find include/nn -name '*.cpp' | tr '\n' ' ') -o /tmp/tg_mut 2>/dev/null; then
    echo "[MUT-$N] $name -> compile error (unexpected for an inert mutation)"
    MISSED=$((MISSED+1)); restore; return
  fi
  if stdbuf -o0 -e0 /tmp/tg_mut >/dev/null 2>&1; then
    echo "[MUT-$N] $name -> INERT as predicted (suite passes; no vacuity)"
    CAUGHT=$((CAUGHT+1))
  else
    echo "[MUT-$N] $name -> !!! DID change behaviour — the 'inert' proof is WRONG !!!"
    MISSED=$((MISSED+1))
  fi
  restore
}

# M1: GAP divides by channels*h*w instead of h*w (the row-mean bug).
run_mutation "M1: GAP over whole row, not per-channel block" \
  's|const double denom = static_cast<double>(h \* w);|const double denom = static_cast<double>(channels * h * w);|'

# M2: the channel loop drops the block offset (indexes from position 0 every time).
run_mutation "M2: GAP loses the c*h*w block offset" \
  's|acc += act_grad\[0\]\[c \* h \* w + s\];|acc += act_grad[0][s];|'

# M3: ReLU dropped from Eq. 2.
run_mutation "M3: no trailing ReLU in Eq. 2" \
  's|cam\[0\]\[s\] = relu ? std::max(0.0, acc) : acc;|cam[0][s] = acc;|'

# M4: ReLU applied per channel BEFORE the weighted sum.
run_mutation "M4: ReLU applied per-channel before weighting" \
  's|acc += alpha\[0\]\[c\] \* act\[0\]\[c \* h \* w + s\];|acc += alpha[0][c] * std::max(0.0, act[0][c * h * w + s]);|'

# M5: the backward walk captures the WRONG tensor (the gradient at the model
# output rather than the one at the target layer). NOTE: this mutation is
# PROVABLY INERT and is listed as a documented negative control, not a live
# test. At i == target_layer + 1 the value entering the call (`grad`) IS
# dL/d(layer i's input) = dL/d(target activation) by the definition of
# backward(), so `next` and `grad` are the same tensor and swapping them cannot
# change any output. Verified by running a probe with and without the mutation:
# alpha = (-0.0383229, -0.0515825, 0.121223), bit-identical. A mutation that
# changes nothing is not a vacuous test — it is a mutation of an unreachable
# distinction. See the M5 note in EXPANSION_QUEUE.md.
run_mutation_expect_inert "M5: backward capture swaps next/grad (INERT by construction)" \
  's|act_grad = next;|act_grad = grad;|'

# M6: upsample uses half-pixel instead of align-corners.
run_mutation "M6: upsample uses half-pixel convention" \
  's|static_cast<double>(y) \* (sh - 1.0) /|((static_cast<double>(y) + 0.5) * sh - 0.5) /|'

# M7: alpha left unnormalised (sum instead of mean).
run_mutation "M7: GAP returns a SUM, not a mean" \
  's|alpha\[0\]\[c\] = acc / denom;|alpha[0][c] = acc;|'

restore
if ! diff -q "$BASE" "$SRC" >/dev/null; then
  echo "FATAL: source not restored after the run — restoring from baseline."
  restore
fi
echo ""
echo "=== Mutations: $N attempted | $CAUGHT caught | $MISSED missed ==="
if [ $MISSED -ne 0 ]; then
  echo "RESULT: FAIL — $MISSED problem(s)"
  exit 1
fi
echo "RESULT: PASS — all mutations caught, source restored"
exit 0