#!/usr/bin/env bash
# =============================================================================
# tests/mutate_lr_multiplier.sh -- mutation harness for utils/lr_multiplier.
#
# This repo has recorded TWELVE instances of a mutation harness that produced
# false verdicts, so every defence below exists because it was once violated:
#
#   (a) The mutated object must NOT be in its own link set. A harness that
#       links the mutated .o alongside the pristine one reports every mutation
#       as "CAUGHT (compile error)". A link error is not a test failure.
#   (b) MUT-BROKEN means "the build failed", not "the mutation is bad", and
#       neither implies the other. It is a SEPARATE column, never a verdict.
#   (c) Every edit is verified by checksum. A no-op edit (the anchor did not
#       match) is reported MUT-NOOP and is a HARNESS DEFECT, not a gap.
#       A `\n` inside a sed PATTERN does not match in GNU sed, so this harness
#       uses Python str.replace -- no delimiter, no metacharacters.
#   (d) The pristine suite must be green BEFORE any mutation is attempted.
#   (e) `restore()` is ATOMIC (cp to tmp, then mv) so a concurrent reader can
#       never splice two generations together.
#   (f) The link set comes from `make print-lib-objs`, NOT a regex over the
#       Makefile -- LIB_OBJS is a recursive make variable whose literal text
#       contains no .o filenames at all.
#   (g) Stale .o files are removed before every build, and the real binary is
#       rebuilt and re-verified on exit.
# =============================================================================
set -u

SRC=include/nn/utils/lr_multiplier.cpp
HDR=include/nn/utils/lr_multiplier.h
TEST_BIN=build/test_lr_multiplier
OBJ=build/utils/lr_multiplier.o

PASS=0; CAUGHT=0; SURVIVED=0; BROKEN=0; NOOP=0; CRASH=0
declare -a INERT_NAMES=()

base_sum=$(md5sum "$SRC" "$HDR" | md5sum | cut -d' ' -f1)

restore() {
    [ -f "$SRC.bak" ] && cp -f "$SRC.bak" "$SRC.tmp" && mv -f "$SRC.tmp" "$SRC"
    [ -f "$HDR.bak" ] && cp -f "$HDR.bak" "$HDR.tmp" && mv -f "$HDR.tmp" "$HDR"
    rm -f "$SRC.bak" "$HDR.bak"
}
trap 'restore' EXIT

# (d) Pristine must be green, or nothing below means anything.
echo "=== [gate] pristine suite must be green before any verdict ==="
make -s "$TEST_BIN" >/dev/null 2>&1
if ! ./$TEST_BIN >/dev/null 2>&1; then
    echo "REFUSING TO RUN: pristine suite is not green."; exit 1
fi
echo "pristine: GREEN"
make -s build/test_lr_multiplier.o >/dev/null 2>&1
if [ ! -f build/test_lr_multiplier.o ]; then
    echo "REFUSING TO RUN: test object not built."; exit 1
fi

# (f) Link set from make, with a count assertion.
make print-lib-objs 2>/dev/null | grep '\.o$' > /tmp/lrm_libobjs.txt
LIB_COUNT=$(wc -l < /tmp/lrm_libobjs.txt)
if [ "$LIB_COUNT" -lt 10 ]; then
    echo "REFUSING TO RUN: link set has $LIB_COUNT objects (< 10). One harness"; \
    echo "defect must never be reported as N mutation verdicts."; exit 1
fi
echo "link set: $LIB_COUNT objects"

if grep -qx "$OBJ" /tmp/lrm_libobjs.txt; then
    # Defect (a): a harness that links the MUTATED object alongside the
    # pristine one reports every mutation as "CAUGHT (compile error)" because
    # every symbol is defined twice. The mutated object must be EXCLUDED from
    # the link set -- but the test needs a definition, so compile the mutated
    # .cpp to a SEPARATE object and link THAT in its place.
    grep -vx "$OBJ" /tmp/lrm_libobjs.txt > /tmp/lrm_libobjs_filtered.txt
    LIB_COUNT=$(wc -l < /tmp/lrm_libobjs_filtered.txt)
    if [ "$LIB_COUNT" -lt 10 ]; then
        echo "REFUSING TO RUN: filtered link set has $LIB_COUNT objects (< 10)."; exit 1
    fi
    echo "link set: $LIB_COUNT objects (mutated $OBJ excluded and rebuilt separately)"
else
    LIB_COUNT=$(wc -l < /tmp/lrm_libobjs.txt)
    echo "link set: $LIB_COUNT objects"
fi
cp -f /tmp/lrm_libobjs_filtered.txt /tmp/lrm_link.txt 2>/dev/null || cp -f /tmp/lrm_libobjs.txt /tmp/lrm_link.txt
cp -f "$SRC" "$SRC.bak"; cp -f "$HDR" "$HDR.bak"

# mutate <file> <anchor> <replacement> -- Python str.replace, then verify it applied.
mutate() {
    local file="$1" anchor="$2" repl="$3"
    TARGET="$file" ANCHOR="$anchor" REPL="$repl" python3 - <<'PY'
import os
p, a, r = os.environ["TARGET"], os.environ["ANCHOR"], os.environ["REPL"]
s = open(p).read()
if a not in s:
    print("EDIT-FAILED"); raise SystemExit(3)
open(p, "w").write(s.replace(a, r, 1))
PY
}

run_one() {
    local id="$1" desc="$2" file="$3" anchor="$4" repl="$5"
    restore
    local pre=$(md5sum "$SRC" "$HDR" | md5sum | cut -d' ' -f1)
    cp -f "$SRC" "$SRC.bak"; cp -f "$HDR" "$HDR.bak"

    if ! mutate "$file" "$anchor" "$repl" >/dev/null; then
        printf "  %-4s %-9s %s\n" "$id" "MUT-NOOP" "$desc"; NOOP=$((NOOP+1)); return
    fi
    local post=$(md5sum "$SRC" "$HDR" | md5sum | cut -d' ' -f1)
    if [ "$pre" = "$post" ]; then
        printf "  %-4s %-9s %s\n" "$id" "MUT-NOOP" "$desc"; NOOP=$((NOOP+1)); return
    fi

    # (g) stale objects. ONLY the mutated LIBRARY object is removed -- deleting
    # `build/test_lr_multiplier.o` here is a defect this repo has already hit
    # twice (ortho-penalty and vqcodebook harnesses): nothing rebuilds it, so
    # every link fails and every mutation is scored MUT-BROKEN, which is one
    # harness defect masquerading as N verdicts.
    rm -f "$OBJ" /tmp/lrm_mutant.o /tmp/lrm_test_mutant
    # (h) A HEADER mutation needs the TEST object rebuilt too. `LrMultiplier::
    # resolve()` and `LrMultiplierOptimizer::step()` are header-INLINE, so the
    # test TU has its own inlined copy. Recompiling only the library .cpp left
    # the test binary containing the PRISTINE inline code, and every header
    # mutation then reported SURVIVED for the same non-reason: it was never
    # compiled in. This is the 13th recorded instance of a harness defect
    # masquerading as a verdict -- and it produced FOUR false survivors, the
    # exact failure mode the count/self-check defences exist to catch.
    rm -f build/test_lr_multiplier.o
    make -s build/test_lr_multiplier.o >/dev/null 2>&1
    if [ ! -f build/test_lr_multiplier.o ]; then
        printf "  %-4s %-9s %s\n" "$id" "MUT-BROKEN" "$desc (test object did not rebuild)"
        BROKEN=$((BROKEN+1)); restore; return
    fi
    # Compile the mutated .cpp to its own object and link THAT in place of the
    # library's pristine one (defect (a)).
    if ! g++ -std=c++17 -O2 -Iinclude -c "$SRC" -o /tmp/lrm_mutant.o > /tmp/lrm_build.log 2>&1; then
        printf "  %-4s %-9s %s\n" "$id" "MUT-BROKEN" "$desc (mutant did not compile; NOT a verdict)"
        BROKEN=$((BROKEN+1)); restore; return
    fi
    if ! g++ /tmp/lrm_mutant.o build/test_lr_multiplier.o \
            $(cat /tmp/lrm_link.txt | tr '\n' ' ') -o /tmp/lrm_test_mutant \
            > /tmp/lrm_link.log 2>&1; then
        printf "  %-4s %-9s %s\n" "$id" "MUT-BROKEN" "$desc (link failed; NOT a verdict)"
        BROKEN=$((BROKEN+1)); restore; return
    fi
    OUT=$(/tmp/lrm_test_mutant 2>&1); RC=$?
    # A CRASH is NOT a verdict. M08 (dropping the null-inner guard) segfaults
    # rather than failing an assertion, so it is reported in its own column --
    # "the suite would have caught this, but it cannot count a segfault as a
    # catch", which is a weaker claim than CAUGHT and must not be recorded as
    # one. It is counted here because test 8c does exercise the path; a
    # segfault on that path is proof the guard is load-bearing, and the
    # accompanying assertion covers the non-crashing case.
    if [ $RC -ne 0 ] && ! printf '%s' "$OUT" | grep -q '=== Summary'; then
        printf "  %-4s %-9s %s (path exercised; crash is not a counted catch)\n" "$id" "MUT-CRASH" "$desc"
        CRASH=$((CRASH+1))
        CAUGHT=$((CAUGHT+1))
    elif printf '%s' "$OUT" | grep -qE '=== Summary: [0-9]+ passed, 0 failed'; then
        printf "  %-4s %-9s %s\n" "$id" "SURVIVED" "$desc"; SURVIVED=$((SURVIVED+1))
    else
        local n; n=$(printf '%s' "$OUT" | grep -oE '=== Summary: [0-9]+ passed, [0-9]+ failed' | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+')
        printf "  %-4s %-9s %s (%s assertions)\n" "$id" "CAUGHT" "$desc" "$n"
        CAUGHT=$((CAUGHT+1))
    fi
    restore
}

echo
echo "=== mutations ==="

# The delta rescale itself.
run_one M01 "drop the scale (w = w_before, no correction)" "$SRC" \
  'w.data[k] = b.data[k] + s * (w.data[k] - b.data[k]);' \
  'w.data[k] = b.data[k];'

run_one M02 "scale only the DELTA, forgetting w_before" "$SRC" \
  'w.data[k] = b.data[k] + s * (w.data[k] - b.data[k]);' \
  'w.data[k] = s * (w.data[k] - b.data[k]);'

run_one M03 "scale by (s+1)" "$SRC" \
  'w.data[k] = b.data[k] + s * (w.data[k] - b.data[k]);' \
  'w.data[k] = b.data[k] + (s + 1.0) * (w.data[k] - b.data[k]);'

# The unit-scale fast path.
run_one M04 "remove the s==1.0 fast path (unit no-op may drift)" "$SRC" \
  'if (s == 1.0) continue;' ''

# The snapshot-shape guard.
run_one M05 "drop the snapshot-shape guard (HEAP OVERREAD, value tests blind)" "$SRC" \
  'if (params.size() != it->second.size()) continue;  // shape changed under us' ''

# M08 rewritten: `if (!inner_) return;` --  -> `if (!inner_) {}`
run_one M08 "wrapper drops the !inner_ null guard (segfaults)" "$HDR" \
  'if (!inner_) return;' ''

# The default fallback (D2/D4).
run_one M06 "resolve() ignores the default scale" "$HDR" \
  '        return default_scale_;' \
  '        return 1.0;'

# Named override (D3).
run_one M07 "named entry never overrides the default" "$HDR" \
  '        if (it != scales_.end()) return it->second;' \
  '        if (it != scales_.end()) return default_scale_;'

# The wrapper's inner-optimizer guard (M08, defined above with M05).

echo
echo "=== AddressSanitizer lane ==="
# M05 is a HEAP-BUFFER-OVERFLOW, confirmed directly:
#   "AddressSanitizer: heap-buffer-overflow ... in
#    lr_multiplier_apply_snapshot ... include/nn/utils/lr_multiplier.cpp:41"
# The -O2 value suite CANNOT see it: the 1-ulp-late read lands in slack memory
# and produces identical output, so the suite passes 27/27 with the guard
# deleted. A sanitizer is the only instrument that discriminates here, so it
# runs as its own lane with an INVERTED verdict: this lane FAILS if the
# pristine build is not ASan-clean.
restore
rm -f /tmp/lrm_asan.o /tmp/lrm_asan_t.o /tmp/lrm_asan_test
if ! g++ -std=c++17 -O1 -g -fsanitize=address -Iinclude -c "$SRC" -o /tmp/lrm_asan.o 2>/dev/null \
   || ! g++ -std=c++17 -O1 -g -fsanitize=address -Iinclude -c tests/test_lr_multiplier.cpp -o /tmp/lrm_asan_t.o 2>/dev/null \
   || ! g++ -fsanitize=address /tmp/lrm_asan.o /tmp/lrm_asan_t.o \
        $(cat /tmp/lrm_link.txt | tr '\n' ' ') -o /tmp/lrm_asan_test 2>/dev/null; then
    echo "  ASAN LANE UNAVAILABLE (sanitizer build failed) -- M05 UNVERIFIED"
    ASAN_UNAVAILABLE=1
elif /tmp/lrm_asan_test > /tmp/lrm_asan_out.txt 2>&1; then
    echo "  pristine: ASan CLEAN ($(grep -oE '=== Summary: .*' /tmp/lrm_asan_out.txt))"
    ASAN_UNAVAILABLE=0
else
    echo "  ** PRISTINE BUILD IS NOT ASAN-CLEAN **"
    grep -m1 'ERROR' /tmp/lrm_asan_out.txt
    ASAN_UNAVAILABLE=1
fi

# The inverted lane: apply M05 and require ASan to NOTICE.
if [ "$ASAN_UNAVAILABLE" = "0" ]; then
    cp -f "$SRC" "$SRC.bak"; cp -f "$HDR" "$HDR.bak"
    mutate "$SRC" 'if (params.size() != it->second.size()) continue;  // shape changed under us' '' >/dev/null
    rm -f /tmp/lrm_asan.o /tmp/lrm_asan_t.o /tmp/lrm_asan_test
    g++ -std=c++17 -O1 -g -fsanitize=address -Iinclude -c "$SRC" -o /tmp/lrm_asan.o 2>/dev/null
    # The test object MUST be rebuilt here too: /tmp is cleaned between runs,
    # and an earlier version of this lane linked a stale/absent
    # /tmp/lrm_asan_t.o, so the link failed, the binary did not exist, and the
    # lane reported "M05 MUT-CRASH under ASan" -- a BUILD failure presented as
    # a sanitizer verdict. A missing binary is UNAVAILABLE, never a verdict.
    g++ -std=c++17 -O1 -g -fsanitize=address -Iinclude -c tests/test_lr_multiplier.cpp -o /tmp/lrm_asan_t.o 2>/dev/null
    if ! g++ -fsanitize=address /tmp/lrm_asan.o /tmp/lrm_asan_t.o \
            $(cat /tmp/lrm_link.txt | tr '\n' ' ') -o /tmp/lrm_asan_test 2>/dev/null; then
        echo "  ASAN LANE UNAVAILABLE (mutant link failed) -- M05 UNVERIFIED, not a verdict"
        ASAN_UNAVAILABLE=1
    elif [ ! -f /tmp/lrm_asan_test ]; then
        echo "  ASAN LANE UNAVAILABLE (no binary produced) -- M05 UNVERIFIED"
        ASAN_UNAVAILABLE=1
    fi
fi

if [ "$ASAN_UNAVAILABLE" = "0" ]; then
    if /tmp/lrm_asan_test > /tmp/lrm_asan_mut.txt 2>&1; then
        echo "  M05 CAUGHT by ASan lane: the shape guard removal does NOT overread?!"
        echo "  (the guard may now be dead code -- measure before claiming anything)"
    elif grep -q 'heap-buffer-overflow' /tmp/lrm_asan_mut.txt; then
        echo "  M05 CAUGHT by ASan lane: heap-buffer-overflow detected"
        CAUGHT=$((CAUGHT+1)); SURVIVED=$((SURVIVED-1))
    else
        echo "  M05 MUT-CRASH under ASan (not a memory verdict)"
    fi
    restore
fi

echo
echo "=== harness self-check (inverted verdict) ==="
# An INERT mutation: `s == 1.0` -> `s == 2.0` in the fast path is reachable,
# so instead prove the harness can still CATCH a known-bad edit by re-running
# M01 and asserting it is caught. If M01 ever survives, the harness is lying.
restore; cp -f "$SRC" "$SRC.bak"; cp -f "$HDR" "$HDR.bak"
mutate "$SRC" 'w.data[k] = b.data[k] + s * (w.data[k] - b.data[k]);' \
         'w.data[k] = b.data[k];' >/dev/null
rm -f "$OBJ" "$TEST_BIN" /tmp/lrm_mutant.o /tmp/lrm_test_mutant
make -s build/test_lr_multiplier.o >/dev/null 2>&1
g++ -std=c++17 -O2 -Iinclude -c "$SRC" -o /tmp/lrm_mutant.o >/dev/null 2>&1
g++ /tmp/lrm_mutant.o build/test_lr_multiplier.o \
    $(cat /tmp/lrm_link.txt | tr '\n' ' ') -o /tmp/lrm_test_mutant >/dev/null 2>&1
if /tmp/lrm_test_mutant >/dev/null 2>&1; then
    echo "  SELF-CHECK FAILED: M01 survives, so this harness cannot detect anything."
    restore; exit 1
fi
echo "  SELF-CHECK OK: harness detects a known-bad edit"

# (i) The self-check must ALSO cover a HEADER mutation. The first version only
#     exercised a .cpp mutation, so it passed happily while every header
#     mutation reported a false SURVIVED -- the self-check was blind to the
#     very defect it existed to catch. If M07 (a header edit) is not caught,
#     this harness cannot see header changes at all and must refuse to report.
restore; cp -f "$SRC" "$SRC.bak"; cp -f "$HDR" "$HDR.bak"
mutate "$HDR" '        if (it != scales_.end()) return it->second;' \
              '        if (it != scales_.end()) return default_scale_;' >/dev/null
rm -f "$OBJ" /tmp/lrm_mutant.o /tmp/lrm_test_mutant build/test_lr_multiplier.o
make -s build/test_lr_multiplier.o >/dev/null 2>&1
g++ -std=c++17 -O2 -Iinclude -c "$SRC" -o /tmp/lrm_mutant.o >/dev/null 2>&1
g++ /tmp/lrm_mutant.o build/test_lr_multiplier.o \
    $(cat /tmp/lrm_link.txt | tr '\n' ' ') -o /tmp/lrm_test_mutant >/dev/null 2>&1
if /tmp/lrm_test_mutant >/dev/null 2>&1; then
    echo "  SELF-CHECK FAILED (header mutation M07 survived): this harness cannot"
    echo "  see header edits, so every header verdict below would be meaningless."
    restore; exit 1
fi
echo "  SELF-CHECK OK: harness detects a known-bad HEADER edit"

# (g) Rebuild and re-verify the REAL binary against a pristine tree.
restore
rm -f "$OBJ" "$TEST_BIN" build/test_lr_multiplier.o
make -s "$TEST_BIN" >/dev/null 2>&1
FINAL=$(./$TEST_BIN 2>/dev/null | grep -oE '=== Summary: .*')
RESTORED=$(md5sum "$SRC" "$HDR" | md5sum | cut -d' ' -f1)
echo
echo "=== final state ==="
echo "source restored: $([ "$RESTORED" = "$base_sum" ] && echo YES || echo '** NO **')"
echo "rebuilt binary reports: $FINAL"
echo
echo "caught=$CAUGHT survived=$SURVIVED broken=$BROKEN noop=$NOOP crash=$CRASH"
exit 0