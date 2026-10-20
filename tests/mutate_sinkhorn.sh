#!/usr/bin/env bash
# =============================================================================
# tests/mutate_sinkhorn.sh -- mutation harness for utils/sinkhorn.{h,cpp}
#
# This harness carries forward EVERY defence the previous sessions in this repo
# needed, because each one was a real defect that produced false verdicts:
#
#   1. `restore()` MUST NEVER DELETE THE BACKUP. It is called once PER MUTATION
#      as well as on EXIT, so a self-cleaning restore works exactly once and
#      every later restore is a silent no-op -- the tree is left holding
#      whatever mutant ran last. (The hutchinson session: the post-harness
#      rebuild reported "23 passed / 17 failed" against a tree nobody edited,
#      and 7 mutants had to be repaired by hand.) Cleanup happens ONCE, on EXIT.
#   2. The link set comes from `make print-lib-objs`, NEVER from regex-parsing
#      the Makefile. `LIB_OBJS` is a RECURSIVE make variable whose literal text
#      contains no `.o` filenames, so a regex yields an empty set and every
#      mutation scores MUT-BROKEN -- one defect reported as N verdicts.
#   3. The MUTATED OBJECT IS EXCLUDED from the link set. A link error is not a
#      test failure.
#   4. BOTH the .cpp and the header are snapshotted and restored ATOMICALLY.
#   5. `rm -f` intermediates before every build (stale .o -> phantom failures).
#   6. EVERY EDIT IS VERIFIED BY CHECKSUM. `MUT-NOOP` means the harness is
#      broken, not that the suite is weak.
#   7. Every throwing call in a statistical test is inside call_guarded, or an
#      inverted guard ABORTS the binary and is reported MUT-CRASH with nothing
#      countable (the muP M13 shape).
#   8. MUT-CRASH gets its OWN column. A missing sanitizer binary is
#      UNAVAILABLE, never a verdict.
#   9. The tree is rebuilt and re-verified ON EXIT before any verdict is
#      reported.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC_CPP="include/nn/utils/sinkhorn.cpp"
SRC_HDR="include/nn/utils/sinkhorn.h"
BIN="build/test_sinkhorn"
OBJ="build/utils/sinkhorn.o"
TOBJ="build/test_sinkhorn.o"

BAK_CPP="$(mktemp)"
BAK_HDR="$(mktemp)"
cp "$SRC_CPP" "$BAK_CPP"
cp "$SRC_HDR" "$BAK_HDR"

restore() {
    # ATOMIC: a plain `cp` truncates the destination first, so a concurrent
    # reader can splice two generations together.
    cp "$BAK_CPP" "$SRC_CPP.tmp" && mv -f "$SRC_CPP.tmp" "$SRC_CPP"
    cp "$BAK_HDR" "$SRC_HDR.tmp" && mv -f "$SRC_HDR.tmp" "$SRC_HDR"
}
cleanup() {
    restore
    rm -f "$BAK_CPP" "$BAK_HDR"
    rm -rf "$SCRATCH"
    # THE POST-HARNESS REBUILD-AND-VERIFY DEFENCE. An earlier session shipped a
    # restored source with a stale mutant still on disk and reported a phantom
    # regression; this is the check that catches it.
    rm -f "$OBJ" "$TOBJ" "$BIN"
    make "$BIN" >/dev/null 2>&1
    if "$BIN" >/dev/null 2>&1; then
        echo ""
        echo "POST-HARNESS: tree restored, rebuilt, and GREEN."
    else
        echo ""
        echo "POST-HARNESS FAILURE: the restored tree does not pass. STOP."
        "$BIN" 2>&1 | tail -20
        exit 1
    fi
}
SCRATCH="$(mktemp -d)"
trap cleanup EXIT INT TERM

checksum() { md5sum "$1" | cut -d' ' -f1; }

# ---------------------------------------------------------------------------
# Link set, from make itself. Assert the count so an empty extraction aborts
# BEFORE any verdict is reported rather than scoring every mutation MUT-BROKEN.
# ---------------------------------------------------------------------------
mapfile -t ALL_OBJS < <(make print-lib-objs)
if [ "${#ALL_OBJS[@]}" -lt 10 ]; then
    echo "ABORT: link set has only ${#ALL_OBJS[@]} objects (need >= 10)."
    exit 1
fi
LINK_OBJS=()
for o in "${ALL_OBJS[@]}"; do
    case "$o" in
        *sinkhorn.o|*test_sinkhorn.o) ;;   # exclude the mutated units
        *) LINK_OBJS+=("$o") ;;
    esac
done
echo "link set: ${#LINK_OBJS[@]} objects (excluded the mutated units)"

# ---------------------------------------------------------------------------
run_suite() {  # -> prints the summary line, or nothing on abort
    stdbuf -o0 "$BIN" 2>/dev/null | grep -E '=== Summary: [0-9]+ passed, [0-9]+ failed ===' \
        | tail -1
}
parse_failed() {   # $1 = summary line -> echo the failure count
    echo "$1" | sed -E 's/.* ([0-9]+) failed ===/\1/'
}

CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0; INERT_CONFIRMED=0
declare -a RESULTS

record() { RESULTS+=("$1|$2|$3"); }

apply_mutation() {  # $1 = file tag (cpp|hdr), $2 = python replacement spec
    python3 - "$1" "$2" <<'PYEOF'
import sys, pathlib
tag, spec = sys.argv[1], sys.argv[2]
path = {"cpp": "include/nn/utils/sinkhorn.cpp",
        "hdr": "include/nn/utils/sinkhorn.h"}[tag]
old, new = spec.split("\x1f")
p = pathlib.Path(path)
s = p.read_text()
if s.count(old) != 1:
    print(f"ANCHOR_COUNT={s.count(old)}")
    sys.exit(3)          # 0 = no-op, >1 = ambiguous
p.write_text(s.replace(old, new))
PYEOF
}

# ---------------------------------------------------------------------------
# SELF-CHECK: prove the harness can produce a CAUGHT verdict at all. A harness
# that has never detected anything is not evidence of a weak suite; it is
# evidence of a broken harness (the curriculum session's verdict-parsing bug
# inverted every result and reported 100% survival against a suite that was
# catching everything).
# ---------------------------------------------------------------------------
echo "=== SELF-CHECK: injecting a known-bad mutation ==="
if apply_mutation cpp "$(printf 'f[i] += eps * (la[i] - lse);\x1ff[i] -= eps * (la[i] - lse);')"; then
    rm -f "$OBJ" "$TOBJ" "$BIN"
    make "$BIN" >/dev/null 2>&1
    line="$(run_suite)"
    if [ -z "$line" ]; then
        echo "SELF-CHECK: MUT-CRASH (no summary line) -- harness counting is suspect"
        SELFCHECK_OK=0
    elif [ "$(parse_failed "$line")" -gt 0 ]; then
        echo "SELF-CHECK: OK -- the suite detects a broken update ($line)"
        SELFCHECK_OK=1
    else
        echo "SELF-CHECK FAILED: the suite passed a deliberately broken solver."
        exit 1
    fi
else
    echo "SELF-CHECK FAILED: could not apply the self-check mutation."
    exit 1
fi
restore
rm -f "$OBJ" "$TOBJ" "$BIN"
make "$BIN" >/dev/null 2>&1
echo "self-check restored; pristine line: $(run_suite)"
echo ""

# ---------------------------------------------------------------------------
# THE MUTATIONS. Each is a mistake the ORACLE'S OWN DERIVATIONS invited -- these
# are not generic edits, they are the specific errors this file's mathematics
# makes easy.
# ---------------------------------------------------------------------------
run_mutation() {  # $1=id $2=file tag $3=spec $4=description
    local id="$1" tag="$2" spec="$3" desc="$4"
    local before after
    before="$(checksum "$SRC_CPP")$(checksum "$SRC_HDR")"

    if ! apply_mutation "$tag" "$spec"; then
        local rc=$?
        restore
        if [ "$rc" -eq 3 ]; then
            echo "  $id MUT-NOOP/AMBIGUOUS -- harness defect, not a verdict: $desc"
            NOOP=$((NOOP+1)); record "$id" "MUT-NOOP" "$desc"; return
        fi
        echo "  $id MUT-NOOP (edit did not apply) -- harness defect: $desc"
        NOOP=$((NOOP+1)); record "$id" "MUT-NOOP" "$desc"; return
    fi

    after="$(checksum "$SRC_CPP")$(checksum "$SRC_HDR")"
    if [ "$before" = "$after" ]; then
        restore
        echo "  $id MUT-NOOP (checksum unchanged) -- harness defect: $desc"
        NOOP=$((NOOP+1)); record "$id" "MUT-NOOP" "$desc"; return
    fi

    rm -f "$OBJ" "$TOBJ" "$BIN"
    if ! make "$BIN" >/dev/null 2>&1; then
        # A BUILD failure is not a verdict. Re-run in isolation to be sure this
        # is the mutation and not a concurrent build race.
        restore; rm -f "$OBJ" "$TOBJ" "$BIN"
        if make "$BIN" >/dev/null 2>&1; then
            echo "  $id MUT-BROKEN transient (build raced, pristine builds fine): $desc"
            BROKEN=$((BROKEN+1)); record "$id" "MUT-BROKEN" "$desc"; return
        fi
        echo "  $id MUT-BROKEN (build failed, persists in isolation): $desc"
        BROKEN=$((BROKEN+1)); record "$id" "MUT-BROKEN" "$desc"; return
    fi

    local line
    line="$(run_suite)"
    if [ -z "$line" ]; then
        echo "  $id CAUGHT (crash, no summary) -- the detector worked, the counting did not: $desc"
        CRASH=$((CRASH+1)); record "$id" "CAUGHT-CRASH" "$desc"; return
    fi
    if [ "$(parse_failed "$line")" -gt 0 ]; then
        local n; n="$(parse_failed "$line")"
        echo "  $id CAUGHT ($n assertions): $desc"
        CAUGHT=$((CAUGHT+1)); record "$id" "CAUGHT($n)" "$desc"; return
    fi
    echo "  $id *** SURVIVED *** $line : $desc"
    SURVIVED=$((SURVIVED+1)); record "$id" "SURVIVED" "$desc"
}

# --- the log-domain update sign (the oracle's first divergence) -------------
run_mutation M01 cpp \
 "$(printf 'f[i] += eps * (la[i] - lse);\x1ff[i] = -eps * (la[i] - lse);')" \
 "update sign negated (doubles error per sweep, NaN in ~40 iters)"

# --- the plan materialisation: forget the a_i factor ------------------------
run_mutation M02 cpp \
 "$(printf 'P(i, j) = ai * std::exp(row[j] - lse);\x1fP(i, j) = ai * std::exp(row[j] - lse) * 2.0;')" \
 "plan rows scaled by 2 (marginals no longer match)"

# --- the self-term coefficient, the constant the oracle got wrong ----------
# The anchor is made UNIQUE by including the leading whitespace of the grad_y
# site, which is indented further. The first run of this harness reported
# MUT-NOOP for both of these because the bare line appears TWICE (once in the
# grad_x branch, once in the grad_y branch) and apply_mutation correctly
# REFUSED an ambiguous anchor rather than silently editing the first match.
# A refusal is the right behaviour; a harness that edits the first match is
# not -- and it is why the count is asserted rather than assumed.
run_mutation M03 cpp "$(printf '            s += (-2.0) * self;\x1f            s += (-1.0) * self;')" \
 "adjoint self-term coefficient -2 -> -1 (grad_x; one endpoint instead of two)"
run_mutation M04 cpp "$(printf '                s += (-2.0) * self;\x1f                s += (-4.0) * self;')" \
 "adjoint self-term coefficient -2 -> -4 (grad_y branch)"

# --- the debiasing terms ---------------------------------------------------
run_mutation M05 cpp "$(printf 'double S = v_xy - 0.5 * v_xx - 0.5 * v_yy;\x1fdouble S = v_xy;')" \
 "debiasing terms removed entirely (a plain OT cost, not a divergence)"
run_mutation M06 cpp "$(printf 'double S = v_xy - 0.5 * v_xx - 0.5 * v_yy;\x1fdouble S = v_xy - 1.0 * v_xx - 1.0 * v_yy;')" \
 "debiasing weights -1/2 -> -1"

# --- the generalised-entropy offset ----------------------------------------
run_mutation M07 cpp \
 "$(printf 'entropy -= p * (std::log(p) - kGeneralizedEntropyOffset);\x1fentropy -= p * std::log(p);')" \
 "generalised-entropy offset dropped (changes the pinned bias magnitude)"
run_mutation M08 cpp \
 "$(printf 'return cost - eps * entropy;\x1freturn cost + eps * entropy;')" \
 "entropy term sign flipped (bias becomes positive)"

# --- the barycentric normalisation -----------------------------------------
run_mutation M09 cpp "$(printf 'out(i, k) = s / mass;\x1fout(i, k) = s;')" \
 "barycentric row normalisation removed"

# --- guards ----------------------------------------------------------------
run_mutation M10 cpp \
 "$(printf 'if (q == i) continue;\x1fif (false) continue;')" \
 "self-term diagonal guard removed (C_xx[i,i]=0 so algebraically inert)"
# The replacement must keep the braces balanced: a first draft emitted
# `if (false)` in place of the condition, which deleted the closing brace and
# reported MUT-BROKEN -- a BUILD verdict, not a mutation verdict, for a
# mutation that was perfectly well formed.
run_mutation M11 cpp \
 "$(printf 'if (C.rows != a.rows || C.cols != b.rows)\x1fif (C.rows != a.rows + 1e9 || C.cols != b.rows)')" \
 "cost/marginal shape guard neutered (braces kept balanced)"
run_mutation M12 cpp \
 "$(printf 'if (!(std::fabs(mass) > 0.0))\x1fif (!(std::fabs(mass) > 0.0 && mass < 1e300))')" \
 "zero-mass barycentric guard neutered (braces kept balanced)"

# --- the max-subtraction in logsumexp --------------------------------------
run_mutation M13 cpp \
 "$(printf 'double mx = row[0];\x1fdouble mx = 0.0;')" \
 "logsumexp max-subtraction neutralised (underflows at small eps)"

# --- the row buffer axis ---------------------------------------------------
run_mutation M14 cpp "$(printf 'std::vector<double> row(m);\x1fstd::vector<double> row(n);')" \
 "row scratch buffer sized from the wrong axis (heap overflow for n>m)"

echo ""
echo "=== ASAN+UBSAN LANE (inverted verdict: it FAILS if the PRISTINE build is not clean) ==="
ASAN_OK="UNAVAILABLE"
rm -rf "$SCRATCH/asan"
mkdir -p "$SCRATCH/asan"
if g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude \
       -c "$SRC_CPP" -o "$SCRATCH/asan/sinkhorn.o" 2>/dev/null &&
   g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude \
       -c tests/test_sinkhorn.cpp -o "$SCRATCH/asan/test.o" 2>/dev/null; then
    # Consistent flags on BOTH TUs: mixing -O1 instrumented and -O2 library
    # objects made inlined constants disagree and produced 11 phantom failures.
    if g++ -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude \
           "$SCRATCH/asan/sinkhorn.o" "$SCRATCH/asan/test.o" \
           "${LINK_OBJS[@]}" -o "$SCRATCH/asan/test_sinkhorn" 2>/dev/null; then
        if ASAN_OPTIONS=detect_leaks=0 "$SCRATCH/asan/test_sinkhorn" \
             >"$SCRATCH/asan/out.txt" 2>&1; then
            ASAN_OK="CLEAN"
            echo "  ASan+UBSan on the PRISTINE tree: CLEAN"
        else
            echo "  ASan+UBSAN FAILURE on the pristine tree:"
            tail -20 "$SCRATCH/asan/out.txt"
        fi
    else
        echo "  ASan link failed -- reporting UNAVAILABLE, never a verdict"
    fi
else
    echo "  ASan compile failed -- reporting UNAVAILABLE, never a verdict"
fi
echo ""
if [ "$ASAN_OK" != "CLEAN" ]; then
    echo "ASAN LANE FAILED (was '$ASAN_OK')."
fi

echo ""
echo "============================================================"
echo " MUTATION SUMMARY: $CAUGHT caught / $SURVIVED survived / $BROKEN broken / $CRASH crash / $NOOP noop"
echo " ASAN pristine build: $ASAN_OK"
echo "============================================================"
for r in "${RESULTS[@]}"; do echo "   $r"; done
