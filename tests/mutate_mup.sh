#!/usr/bin/env bash
# Mutation harness for tests/test_mup.cpp.
#
# Carries every defence this repo's sessions have had to learn the hard way,
# because the failure mode it guards against is one harness defect reported as
# N mutation verdicts -- which would "certify" a suite that catches nothing.
#
#   1. Link set from `make print-lib-objs`, COUNTED, abort below 10 objects.
#      `LIB_OBJS` is a RECURSIVE make variable (`:=` over a `$(LIB_SRCS:...)`
#      substitution), so its literal Makefile text contains ZERO `.o`
#      filenames. Two earlier harnesses in this repo parsed it with a regex,
#      got an empty set, and scored all their real mutations MUT-BROKEN.
#   2. The mutated TU is EXCLUDED from the link set. Linking it twice
#      double-defines every symbol and reports every mutation "CAUGHT
#      (compile error)" -- a link error is not a test failure.
#   3. Every edit is applied with a Python literal `str.replace` (no sed
#      delimiter, no `\n`-in-pattern, no BRE-vs-ERE trap) and VERIFIED with a
#      checksum before any verdict is produced. `MUT-NOOP` is correct for a
#      no-op but produces no evidence.
#   4. BOTH `mup.cpp` AND `mup.h` are snapshotted and restored, because a
#      mutation can live in either. An earlier session snapshotted only the
#      .cpp while mutating the header, leaked it into the tree, and printed a
#      false "source restored: YES".
#   5. Restore is ATOMIC (`cp base tmp && mv -f tmp src`); a plain `cp`
#      truncates the destination first and a concurrent reader can splice two
#      generations together.
#   6. `MUT-BROKEN` (build failed) and `MUT-CRASH` (the binary aborted) are
#      their OWN columns. Neither is ever scored as caught or survived: a
#      crash produces no [FAIL] line, so a [FAIL]-counting verdict reads it as
#      a pass.
#   7. `rm -f` the intermediates before EVERY build. A stale .o from the
#      previous mutation produced a build failure that did not reproduce in
#      isolation.
#   8. The pristine suite must be GREEN before a single verdict is reported.
#   9. On exit, the REAL binary is rebuilt and re-run against a pristine tree,
#      because a harness that leaves the last mutant in build/ reports a
#      phantom regression in code that has not changed.

set -uo pipefail
cd "$(dirname "$0")/.."

SRC=include/nn/utils/mup.cpp
HDR=include/nn/utils/mup.h
BIN=build/test_mup

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

CAUGHT=0; SURVIVED=0; BROKEN=0; CRASH=0; NOOP=0; INERT_OK=0
declare -a ROWS=()

say() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------- link set
LIB_LIST=$(make print-lib-objs 2>/dev/null | grep '\.o$')
N_LIB=$(printf '%s\n' "$LIB_LIST" | grep -c .)
if [ "$N_LIB" -lt 10 ]; then
    say "HARNESS DEFECT: link set has $N_LIB objects (<10). Aborting before any verdict."
    exit 1
fi
# Drop the object under test -- linking it twice double-defines every symbol.
LIB_LIST=$(printf '%s\n' "$LIB_LIST" | grep -v "utils/mup.o")
say "link set: $(( $(printf '%s\n' "$LIB_LIST" | grep -c .) )) objects (mutated TU excluded)"

# ---------------------------------------------------------------- snapshots
cp "$SRC" "$TMP/mup.cpp.base"
cp "$HDR" "$TMP/mup.h.base"

restore() {
    cp "$TMP/mup.cpp.base" "$TMP/mup.cpp.new" && mv -f "$TMP/mup.cpp.new" "$SRC"
    cp "$TMP/mup.h.base"   "$TMP/mup.h.new"   && mv -f "$TMP/mup.h.new"   "$HDR"
}

sums() { md5sum "$SRC" "$HDR" | awk '{print $1}' | tr '\n' ' '; }
BASE_SUMS=$(sums)

# ---------------------------------------------------------------- build one
# $1 = label. Prints the [FAIL] count, or a marker word.
build_and_run() {
    rm -f "$BIN" build/test_mup.o build/utils/mup.o
    if ! make "$BIN" >"$TMP/build.log" 2>&1; then
        echo "MUT-BROKEN"
        return
    fi
    # A mutation that merely introduces -Wunused-variable is a build verdict,
    # not a mutation verdict, so the project flags are already fatal above.
    local out rc
    out=$(./"$BIN" 2>&1); rc=$?
    if [ $rc -ge 128 ] || printf '%s' "$out" | grep -qE 'terminate called|Segmentation fault|AddressSanitizer'; then
        echo "MUT-CRASH"
        return
    fi
    printf '%s' "$out" | grep -c '^\[FAIL\]' || true
}

# ---------------------------------------------------------------- baseline
restore
BASE_FAILS=$(build_and_run)
if [ "$BASE_FAILS" != "0" ]; then
    say "HARNESS DEFECT: pristine suite is not green ($BASE_FAILS failures). Aborting."
    exit 1
fi
say "pristine suite: green"
say ""

# ---------------------------------------------------------------- mutations
# format: TAG | FILE | OLD | NEW | EXPECT(caught|inert)
#
# Each OLD is a verbatim substring of the file; the harness refuses to report
# a verdict for an edit that does not apply.
run_mut() {
    local tag="$1" file="$2" old="$3" new="$4" expect="${5:-caught}"
    restore
    python3 - "$file" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if s.count(old) != 1:
    sys.exit(3)          # zero or ambiguous anchor -> MUT-NOOP, no verdict
open(path, 'w').write(s.replace(old, new, 1))
PY
    local rc=$?
    if [ $rc -eq 3 ]; then
        say "MUT-NOOP  $tag (anchor not unique in $file -- NO EVIDENCE)"
        NOOP=$((NOOP + 1)); restore; return
    fi
    local now; now=$(sums)
    if [ "$now" = "$BASE_SUMS" ]; then
        say "MUT-NOOP  $tag (edit did not change the tree)"
        NOOP=$((NOOP + 1)); restore; return
    fi
    local fails; fails=$(build_and_run)
    restore
    local row
    case "$fails" in
        MUT-BROKEN) BROKEN=$((BROKEN+1));  row=$(printf '%-52s %s' "$tag" MUT-BROKEN) ;;
        MUT-CRASH)  CRASH=$((CRASH+1));   row=$(printf '%-52s %s' "$tag" MUT-CRASH) ;;
        0)          SURVIVED=$((SURVIVED+1)); row=$(printf '%-52s %s' "$tag" "SURVIVED") ;;
        *)          CAUGHT=$((CAUGHT+1)); row=$(printf '%-52s %s' "$tag" "CAUGHT ($fails) failed)") ;;
    esac
    ROWS+=("$row")
    printf '%s\n' "$row"
}

# ---- M01: (D1) the one squared cell in Table 3 becomes unsquared.
run_mut "M01 output init 1/fan_in^2 -> 1/fan_in" "$SRC" \
    'return 1.0 / (n * n);' 'return 1.0 / n;'

# ---- M02: (D3) the asymmetry -- Adam input row stops being invariant.
run_mut "M02 adam input LR 1 -> 1/r" "$SRC" \
    '        case MupRole::Input:
            return 1.0;
        case MupRole::Hidden:
        case MupRole::Output:
            return check_finite(1.0 / r, "adam_lr_rel");' \
    '        case MupRole::Input:
        case MupRole::Hidden:
        case MupRole::Output:
            return check_finite(1.0 / r, "adam_lr_rel");'

# ---- M03: (D3) and the SGD input row stops scaling.
run_mut "M03 sgd input LR r -> 1" "$SRC" \
    '        case MupRole::Input:
            return check_finite(r, "sgd_lr_rel");' \
    '        case MupRole::Input:
            return 1.0;'

# ---- M04: (D5) the attention exponent -- sqrt(d0/d) collapses to 1/d.
run_mut "M04 attn sqrt(d0/d) -> d0/d" "$SRC" \
    'return check_finite(alpha_attn * std::sqrt(d0 / d), "attn_scale");' \
    'return check_finite(alpha_attn * d0 / d, "attn_scale");'

# ---- M05: (D5) the other exponent -- sqrt(d0/d) collapses to sqrt(d0)/d.
run_mut "M05 attn sqrt(d0/d) -> sqrt(d0)/d" "$SRC" \
    'return check_finite(alpha_attn * std::sqrt(d0 / d), "attn_scale");' \
    'return check_finite(alpha_attn * std::sqrt(d0) / d, "attn_scale");'

# ---- M06: (D3) hidden SGD LR row stops being invariant.
run_mut "M06 sgd hidden LR 1 -> 1/r" "$SRC" \
    '        case MupRole::Hidden:
            return 1.0;
        // 1/fan_in => 1/r.' \
    '        case MupRole::Hidden:
            return check_finite(1.0 / r, "sgd_lr_rel");
        // 1/fan_in => 1/r.'

# ---- M07: the init-var direction inverts (a real bug this session FIXED).
run_mut "M07 init_var_rel 1/r -> r" "$SRC" \
    '        case MupRole::Input:
        case MupRole::Hidden:
            return check_finite(1.0 / r, "init_var_rel");' \
    '        case MupRole::Input:
        case MupRole::Hidden:
            return check_finite(r, "init_var_rel");'

# ---- M08: (D6) biases switch from the input row to the output row.
run_mut "M08 bias adam LR input row -> output row" "$SRC" \
    'double mup_adam_lr_bias(size_t /*fan_out*/) {
    return mup_adam_lr_input(1, 1);
}' \
    'double mup_adam_lr_bias(size_t fan_out) {
    return mup_adam_lr_output(fan_out);
}'

# ---- M09: (D7) the zero guard is removed from one entry point.
run_mut "M09 drop require_nonzero in init_var_hidden" "$SRC" \
    'double mup_init_var_hidden(size_t fan_in) {
    require_nonzero(fan_in, "fan_in");' \
    'double mup_init_var_hidden(size_t fan_in) {'

# ---- M10: base_width no longer participates (D4 broken).
run_mut "M10 base_width ignored in width_ratio" "$SRC" \
    'return static_cast<double>(s.width) / static_cast<double>(s.base_width);' \
    'return static_cast<double>(s.width) / static_cast<double>(s.width);'

# ---- M11: the D2 output-SGD-LR cell keys off the wrong power.
run_mut "M11 sgd output LR 1/fan_in -> 1" "$SRC" \
    'double mup_sgd_lr_output(size_t fan_in) {
    require_nonzero(fan_in, "fan_in");
    return 1.0 / static_cast<double>(fan_in);
}' \
    'double mup_sgd_lr_output(size_t fan_in) {
    require_nonzero(fan_in, "fan_in");
    return 1.0;
}'

# ---- M12: alpha_attn stops being linear in alpha.
run_mut "M12 attn alpha multiplicative -> squared" "$SRC" \
    'return check_finite(alpha_attn * std::sqrt(d0 / d), "attn_scale");' \
    'return check_finite(alpha_attn * alpha_attn * std::sqrt(d0 / d), "attn_scale");'

# ---- M13: the role-string dispatch misroutes "bias" to "hidden". First run
#      reported MUT-CRASH, not a verdict: the suite's own 4-role fixture
#      triggered an UNCAUGHT throw from its setup and killed the process
#      before any assertion ran. The suite now checks each role ALONE, so a
#      misrouted role throws inside check_throws and is reported as a failure.
run_mut "M13 role dispatch bias -> hidden" "$SRC" \
    '        } else if (r == "bias") {' \
    '        } else if (r == "hidden") {'

# ---- M14: update_ratio stops throwing on a zero activation norm.
run_mut "M14 drop update_ratio zero guard" "$SRC" \
    '    if (activation_norm == 0.0) {
        throw std::invalid_argument("mup: activation_norm must be nonzero");
    }' \
    '    if (false) {
        throw std::invalid_argument("mup: activation_norm must be nonzero");
    }'

# ---------------------------------------------------------------- summary
restore
say ""
say "======================================================"
for r in "${ROWS[@]}"; do say "  $r"; done
say "======================================================"
say "caught   : $CAUGHT"
say "survived : $SURVIVED"
say "broken   : $BROKEN   (build failed -- NOT a verdict)"
say "crash    : $CRASH   (aborted -- NOT a verdict)"
say "noop     : $NOOP   (edit never applied -- NOT a verdict)"

# Rebuild the REAL binary against a pristine tree and re-verify, so the next
# run cannot read a stale mutant as a regression.
rm -f "$BIN" build/test_mup.o build/utils/mup.o
if make "$BIN" >"$TMP/rebuild.log" 2>&1; then
    say ""
    say "post-run rebuild on pristine tree: $(./"$BIN" | tail -1)"
else
    say "POST-RUN REBUILD FAILED -- see below"
    tail -20 "$TMP/rebuild.log"
fi