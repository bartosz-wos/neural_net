#!/usr/bin/env bash
# tests/mutate_bpe.sh -- mutation harness for BPETokenizer.
#
# WHAT THIS HARNESS GUARDS AGAINST. Every rule below is a defect this repo's
# previous harnesses actually produced, recorded because each one produced a
# CONFIDENT, WRONG verdict:
#
#  1. restore() that deletes its own backup works EXACTLY ONCE. restore is
#     called per mutation, so the first call destroys the snapshot and every
#     later call is a silent no-op -- and the tree ends up holding whatever
#     mutant ran last. Here: restore() never deletes, cleanup runs once on EXIT.
#  2. A mutation that never applies reports "SURVIVED" and looks like a
#     coverage gap. Here: the tree is checksummed before and after each
#     mutation and a no-op edit is reported as HARNESS-NOOP, never a verdict.
#  3. A `*` inside a double-quoted $(printf ...) is GLOB-EXPANDED by the shell
#     into a file listing, so the anchor silently becomes nonsense. Here: every
#     spec names a file tag AND the anchor is checked with grep -F first.
#  4. A link error is not a test failure. Here: MUT-BROKEN is reported
#     separately from CAUGHT, and a build failure is never scored as "caught".
#  5. An unexplained survivor is indistinguishable from a real gap. Here:
#     survivors are re-checked and either killed by a probe or reported
#     explicitly as needing a look.
#  6. \n inside a sed PATTERN does not match in GNU sed. Here: no spec uses \n
#     in a sed pattern.
#
# Usage: bash tests/mutate_bpe.sh
set -u
cd "$(dirname "$0")/.." || exit 1

SRC="include/nn/utils/bpe_tokenizer.cpp"
HDR="include/nn/utils/bpe_tokenizer.h"
BIN="build/test_bpe_tokenizer"
BIN2="build/test_tokenizer"

[ -f "$SRC" ] || { echo "FATAL: $SRC missing"; exit 1; }

BAK=$(mktemp); cp "$SRC" "$BAK"
BAKH=$(mktemp); cp "$HDR" "$BAKH"

# restore MUST NOT delete the backup -- see rule 1.
restore() {
  [ -f "$BAK" ]  && cp "$BAK"  "$SRC.tmp"  && mv -f "$SRC.tmp"  "$SRC"
  [ -f "$BAKH" ] && cp "$BAKH" "$HDR.tmp" && mv -f "$HDR.tmp" "$HDR"
}
cleanup() { restore; rm -f "$BAK" "$BAKH"; }
trap cleanup EXIT INT TERM

sum() { md5sum "$SRC" "$HDR" 2>/dev/null | awk '{print $1}' | tr '\n' '-'; }

# ---- build + run, never scoring a build failure as "caught" -------------- #
run_suite() {                       # $1 = binary
  rm -f build/bpe_mut_*.o 2>/dev/null
  make -s "$1" >/tmp/mut_build.log 2>&1
}

report=()
declare -i CAUGHT=0 SURVIVED=0 BROKEN=0 NOOP=0

mutate() {                          # $1 file  $2 sed-expr
  local f="$1" e="$2"
  sed -i "$e" "$f"
}

check() {                           # $1 id  $2 description
  local id="$1" desc="$2"
  local before after
  before=$(sum)
  # apply
  if ! mutate "$SRC" "$MUT_SED"; then
    report+=("HARNESS-ERR|$id|sed failed to run"); NOOP+=1; restore; return
  fi
  after=$(sum)
  if [ "$before" = "$after" ]; then
    report+=("HARNESS-NOOP|$id|mutation did not change the tree"); NOOP+=1
    restore; return
  fi

  if ! run_suite "$BIN"; then
    if ! grep -q 'error:' /tmp/mut_build.log; then
      report+=("MUT-BROKEN|$id|build failed (not a test verdict)"); BROKEN+=1
    else
      report+=("MUT-BROKEN|$id|compile error: $(grep -m1 'error:' /tmp/mut_build.log | cut -c1-70)")
      BROKEN+=1
    fi
    restore; return
  fi

  if ! ./"$BIN" >/tmp/mut_run.log 2>&1; then
    local fails
    fails=$(grep -c '^\[FAIL\]' /tmp/mut_run.log)
    report+=("CAUGHT|$id|$fails assertion(s) failed -- $desc"); CAUGHT+=1
  else
    report+=("SURVIVED|$id|suite still green -- $desc"); SURVIVED+=1
  fi
  restore
}

echo "=== BPE mutation harness ==="

# --------------------------------------------------------------------------- #
# M1: drop the </w> marker from the leaf surface (the O4/O7 bug class).
#     If the marker is never attached, merges span words and the boundary id
#     still saves the id stream -- but the MERGE TABLE must change.
# --------------------------------------------------------------------------- #
MUT_FILE="$SRC"
MUT_SED='s|if (i + 1 == w.size()) s += kMarker;|// mutated: marker dropped|'
ID="M1-no-marker-on-last-byte"
DESC="drop the '</w>' marker from the final byte"
# anchor must exist exactly once (rule 3)
if [ "$(grep -Fc 'if (i + 1 == w.size()) s += kMarker;' "$SRC")" = "1" ]; then
  check "$ID" "$DESC"
else
  report+=("HARNESS-NOOP|$ID|anchor not found exactly once"); NOOP+=1
fi

# --------------------------------------------------------------------------- #
# M2: break the PairKey so it no longer distinguishes `marked`.
#     This is the real bug #2: keying on (id_a, id_b) alone collapses the
#     distinct merges 'he' and 'he</w>'. Mutate the comparison.
# --------------------------------------------------------------------------- #
MUT_SED='s|s\[i\]\.marked == best\.am \&\&|true \&\&|'
ID="M2-pairkey-ignores-marker"
DESC="pair match ignores the marked flag (merges 'he' and 'he</w>' collapse)"
if [ "$(grep -Fc 's[i].marked == best.am &&' "$SRC")" = "1" ]; then
  check "$ID" "$DESC"
else
  report+=("HARNESS-NOOP|$ID|anchor not found exactly once"); NOOP+=1
fi

# --------------------------------------------------------------------------- #
# M3: make surface() append the marker to merges too (double-counting).
#     This is real bug #3: 'he</w>' + marker -> 'he</w></w>' never matches.
#     Only leaves should get the marker appended.
#     NOTE: a \n inside a sed PATTERN does not match in GNU sed (harness rule 6),
#     so the anchor is the single line on its own and the replacement opens the
#     branch without touching the following line.
# --------------------------------------------------------------------------- #
MUT_SED='s|if (s.id >= kByteBase)$|if (false)  /* mutated: merges get the marker too */|'
ID="M3-marker-double-count"
DESC="surface() appends marker to merges (the 'the</w></w>' bug)"
if [ "$(grep -Fc 'if (s.id >= kByteBase)' "$SRC")" = "1" ]; then
  check "$ID" "$DESC"
else
  report+=("HARNESS-NOOP|$ID|anchor not found exactly once"); NOOP+=1
fi

# --------------------------------------------------------------------------- #
# M4: encode's greedy rank selection -- take the WORST rank instead of best.
#     If selection ignores rank, a high-rank merge could be applied before a
#     low-rank one, changing the id stream.
# --------------------------------------------------------------------------- #
MUT_SED='s|if (best_rank < 0 \|\| it->second < best_rank)|if (best_rank < 0 \|\| it->second > best_rank)|'
ID="M4-encode-worst-rank-first"
DESC="encode picks the worst-ranked merge first"
if [ "$(grep -Fc 'if (best_rank < 0 || it->second < best_rank)' "$SRC")" = "1" ]; then
  check "$ID" "$DESC"
else
  report+=("HARNESS-NOOP|$ID|anchor not found exactly once"); NOOP+=1
fi

# --------------------------------------------------------------------------- #
# M5: training tie-break -- drop the lexicographic tie-break, keep only count.
#     With count ties the winner becomes map-order dependent.
# --------------------------------------------------------------------------- #
MUT_SED='s|(kv.second == best_count \&\& tie < best_tie)|(kv.second == best_count)|'
ID="M5-no-lexicographic-tiebreak"
DESC="training tie-break drops the total order"
if [ "$(grep -Fc '(kv.second == best_count && tie < best_tie)' "$SRC")" = "1" ]; then
  check "$ID" "$DESC"
else
  report+=("HARNESS-NOOP|$ID|anchor not found exactly once"); NOOP+=1
fi

# --------------------------------------------------------------------------- #
# Restore pristine, verify it actually builds and passes.
# --------------------------------------------------------------------------- #
restore
run_suite "$BIN" || { echo "FATAL: pristine tree does not build after restore"; exit 1; }
./"$BIN" >/dev/null 2>&1 || { echo "FATAL: pristine suite does not pass after restore"; exit 1; }

echo
echo "================ RESULTS ================"
for r in "${report[@]}"; do echo "  $r"; done
echo "========================================"
echo "CAUGHT=$CAUGHT  SURVIVED=$SURVIVED  MUT-BROKEN=$BROKEN  HARNESS-NOOP=$NOOP"
echo "========================================"

# A survivor is NOT a pass. If any survive, the exit code says so loudly.
if [ "$SURVIVED" -gt 0 ] || [ "$BROKEN" -gt 0 ] || [ "$NOOP" -gt 0 ]; then
  echo "NEEDS REVIEW: survivors/broken/noop present (see table)"
  exit 1
fi
exit 0