#!/usr/bin/env bash
# =============================================================================
# tests/mutate_stack_cached_decode.sh -- the source-mutation harness for the
# STACK-LEVEL CACHED DECODE.
#
# PURPOSE: prove that the C++ suite CATCHES the defects the feature exists to
# prevent. A green suite proves nothing on its own; the mutation has to fail.
#
#     bash tests/mutate_stack_cached_decode.sh
#
# THE MUTATION THAT MATTERS MOST
# =============================================================================
#
# SM1 gives EVERY BLOCK THE SAME CACHE SLOT. It does not throw, does not
# crash, and does not look wrong: the stack would silently attend to the
# FIRST block's keys and values from EVERY block. The block-level
# cached-decode test could not catch this, because that path only ever had
# ONE block -- where "one slot" and "one slot per block" are the same
# thing. The stack's whole reason to exist is to make the distinction.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
# =============================================================================
#
#   1. THE LINK SET COMES FROM `make print-lib-objs`, NEVER FROM A REGEX OVER
#      THE MAKEFILE. `LIB_OBJS` is a RECURSIVE make variable whose literal
#      text contains a make reference rather than filenames, so a regex
#      yields ZERO objects and every mutation would score a false
#      MUT-BROKEN. The object count is asserted to be >= 10.
#   2. THE MUTATED TRANSLATION UNIT IS COMPILED SEPARATELY AND SWAPPED IN.
#      Linking it together with the library's own copy double-defines every
#      symbol and reports "caught" for every mutation -- a link error is NOT
#      a test failure. The mutated object's path is dropped from the link
#      set.
#   3. THE INTERMEDIATES ARE REMOVED BEFORE EVERY BUILD. A stale .o from the
#      previous mutation makes a mutant report a build failure that does not
#      reproduce in isolation.
#   4. restore() IS ATOMIC AND NEVER DELETES THE BACKUP. Cleanup happens
#      exactly once, on EXIT. A restore() that deletes its own backup works
#      EXACTLY ONCE, and every later restore is a silent no-op -- which
#      leaves a mutant in the tree.
#   5. A STALE SOURCE IS A HARNESS DEFECT, NOT A VERDICT: the source is
#      checksummed against the backup after every edit.
#   6. THE ANCHOR MUST BE UNIQUE. A line that appears twice makes the edit a
#      coin flip, and a coin flip is not a verdict.
#   7. THE EDITS ARE DONE WITH A PYTHON SCRIPT, not perl/sed: this repo has
#      been bitten by perl's s/// delimiter (the anchors are full of "/")
#      and by sed interpreting a newline inside a pattern.
#   8. THIS HARNESS MUTATES `transformer_stack.cpp` ONLY, so it does not
#      conflict with tests/mutate_block_cached_decode.sh, which mutates
#      `transformer.cpp`. They still share `build/*.o`, so only one of them
#      may run at a time.
#
# =============================================================================
# THE VERDICT CLASSES
# =============================================================================
#
#   CAUGHT     the suite FAILED. This is the only good outcome.
#   SURVIVED   the suite PASSED. This is a hole in the SUITE.
#   MUT-BROKEN the BUILD failed. Not a verdict about the suite.
#   MUT-CRASH  the binary died (rc >= 128) with nothing reported. Not a
#               verdict about the suite.
#   MUT-NOOP   the mutation did not apply. A harness defect.
#   MUT-ANCHOR the anchor was not unique. A harness defect.
# =============================================================================

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO_ROOT}/include/nn/layers/architectures/transformer_stack.cpp"
BUILD_DIR="${REPO_ROOT}/build"
MUT_BUILD="${REPO_ROOT}/build/mutation/stack_cached_decode"
TEST_SRC="${REPO_ROOT}/tests/test_stack_cached_decode.cpp"

CAUGHT=0
SURVIVED=0
BROKEN=0
CRASHED=0
NOOP=0
ANCHOR_BAD=0

# --- the backup. restore() NEVER deletes it; cleanup does, on EXIT. --------
BACKUP="$(mktemp)"
cp "${TARGET}" "${BACKUP}"

restore() {
    # ATOMIC: cp to a temp then mv, so a concurrent reader never sees a
    # half-written source.
    cp "${BACKUP}" "${TARGET}.tmp" && mv -f "${TARGET}.tmp" "${TARGET}"
}

cleanup() {
    restore
    rm -f "${BACKUP}"
    rm -rf "${MUT_BUILD}"
}
trap cleanup EXIT INT TERM

# =============================================================================
# The mutation table.
#   id | description | python search | python replacement
# =============================================================================
read -r -d '' MUTATIONS <<'PYEOF'
SM1|EVERY BLOCK IS GIVEN THE SAME CACHE SLOT (slot 0)|        blocks_[i].set_kv_cache(kv_cache_, static_cast<int>(i));|        blocks_[i].set_kv_cache(kv_cache_, static_cast<int>(0));
SM2|the blocks are chained in REVERSE order|    Tensor h = input;\n    for (size_t i = 0; i < blocks_.size(); ++i) {\n        h = blocks_[i].forward_cached(h);\n    }|    Tensor h = input;\n    for (size_t i = blocks_.size(); i-- > 0;) {\n        h = blocks_[i].forward_cached(h);\n    }
SM3|the cached path uses the ORDINARY forward (no cache at all)|        h = blocks_[i].forward_cached(h);|        h = blocks_[i].forward(h);
SM4|the cached path SKIPS the last block|        h = blocks_[i].forward_cached(h);|        h = (i + 1 < blocks_.size()) ? blocks_[i].forward_cached(h) : h;
SM5|the backward chains the blocks in the FORWARD order|    Tensor g = grad_output;\n    for (size_t i = blocks_.size(); i-- > 0;) {\n        g = blocks_[i].backward(g, learning_rate);\n    }|    Tensor g = grad_output;\n    for (size_t i = 0; i < blocks_.size(); ++i) {\n        g = blocks_[i].backward(g, learning_rate);\n    }
SM6|the slots are NOT reserved up front|        kv_cache_->reserve(static_cast<int>(i), static_cast<int>(d_model_));\n|        
SM7|reset_cache() does nothing|void TransformerStack::reset_cache() {\n    if (kv_cache_) kv_cache_->reset_all();\n}|void TransformerStack::reset_cache() {\n}
SM8|the constructor's num_layers guard is dropped|    if (num_layers == 0) {\n        throw std::invalid_argument(\n            "TransformerStack: num_layers must be positive, got 0");\n    }\n|    if (false) {\n        throw std::invalid_argument(\n            "TransformerStack: num_layers must be positive, got 0");\n    }\n
SM9|the "no KV cache" message changes|            "TransformerStack::forward_cached: no KV cache is attached; "\n            "call set_kv_cache() first");|            "TransformerStack::forward_cached: the cache is missing");
SM10|the stack's own "no KV cache" guard is dropped|    if (!kv_cache_) {\n        throw std::runtime_error(\n            "TransformerStack::forward_cached: no KV cache is attached; "\n            "call set_kv_cache() first");\n    }\n|    if (false) {\n        throw std::runtime_error(\n            "TransformerStack::forward_cached: no KV cache is attached; "\n            "call set_kv_cache() first");\n    }\n
SM11|set_kv_cache does not hand the cache to the blocks|    kv_cache_ = cache;\n    attach_cache_to_blocks();|    kv_cache_ = cache;
SM12|the constructor does not attach its own cache|    kv_cache_ = std::make_shared<nn::utils::KVCache>(static_cast<int>(max_seq_len));\n    attach_cache_to_blocks();|    kv_cache_ = std::make_shared<nn::utils::KVCache>(static_cast<int>(max_seq_len));
PYEOF

# =============================================================================
# The link set, from `make print-lib-objs` -- NEVER from a regex over the
# Makefile.
# =============================================================================
LIB_OBJS="$(cd "${REPO_ROOT}" && make -s print-lib-objs | tr '\n' ' ')"
LIB_COUNT="$(echo "${LIB_OBJS}" | wc -w)"
if [ "${LIB_COUNT}" -lt 10 ]; then
    echo "MUT-HARNESS-BROKEN: the link set has only ${LIB_COUNT} objects (must be >= 10)"
    exit 2
fi
echo "link set: ${LIB_COUNT} objects"

mkdir -p "${MUT_BUILD}"

# =============================================================================
# The main loop.
# =============================================================================
while IFS='|' read -r mid desc search repl; do
    [ -z "${mid}" ] && continue

    # --- restore FIRST. Every iteration starts from the pristine source.
    restore

    # --- apply the mutation with python (no shell delimiter hazards).
    python3 - "${TARGET}" "${search}" "${repl}" <<'PYEOF'
import sys
path, search, repl = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
search = search.replace("\\n", "\n")
repl = repl.replace("\\n", "\n")
count = src.count(search)
if count != 1:
    sys.stderr.write("ANCHOR count=%d\n" % count)
    sys.exit(3)
open(path, "w").write(src.replace(search, repl))
PYEOF
    rc_edit=$?
    if [ $rc_edit -eq 3 ]; then
        echo "${mid} MUT-ANCHOR  the anchor is not unique -- harness defect, not a verdict"
        ANCHOR_BAD=$((ANCHOR_BAD + 1))
        continue
    elif [ $rc_edit -ne 0 ]; then
        echo "${mid} MUT-NOOP  the edit failed -- harness defect, not a verdict"
        NOOP=$((NOOP + 1))
        continue
    fi

    # --- checksum: a mutation that did not apply is a harness defect.
    if cmp -s "${TARGET}" "${BACKUP}"; then
        echo "${mid} MUT-NOOP  the source is unchanged -- harness defect"
        NOOP=$((NOOP + 1))
        continue
    fi

    # --- build. The mutated TU is compiled separately and swapped in; the
    #     library's copy of it is dropped from the link set.
    LINK_OBJS=""
    for o in ${LIB_OBJS}; do
        case "$o" in
            */layers/architectures/transformer_stack.o) ;;
            *) LINK_OBJS="${LINK_OBJS} $o" ;;
        esac
    done
    MUT_OBJ="${MUT_BUILD}/mutated.o"
    rm -f "${MUT_OBJ}" "${MUT_BUILD}/test_stack_cached_decode" "${MUT_BUILD}/test_stack_cached_decode.o"
    if ! g++ -std=c++17 -O2 -Wall -Wextra -march=native -I"${REPO_ROOT}/include" \
            -c "${TARGET}" -o "${MUT_OBJ}" 2> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the mutated TU did not compile -- a BUILD verdict, not a coverage one"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ -std=c++17 -O2 -I"${REPO_ROOT}/include" \
            -c "${TEST_SRC}" -o "${MUT_BUILD}/test_stack_cached_decode.o" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the test did not compile -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ "${MUT_BUILD}/test_stack_cached_decode.o" "${MUT_OBJ}" ${LINK_OBJS} \
            -o "${MUT_BUILD}/test_stack_cached_decode" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the link failed -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi

    # --- run the suite.
    "${MUT_BUILD}/test_stack_cached_decode" > "${MUT_BUILD}/run.log" 2>&1
    rc=$?
    n_fail=$(grep -c '\[FAIL\]' "${MUT_BUILD}/run.log" 2>/dev/null || true)
    n_fail=${n_fail:-0}
    first=$(grep -m1 '\[FAIL\]' "${MUT_BUILD}/run.log" | cut -c1-140)

    if [ $rc -eq 0 ]; then
        echo "${mid} SURVIVED  ${desc}"
        SURVIVED=$((SURVIVED + 1))
    elif [ $rc -ge 128 ] && [ "${n_fail}" -eq 0 ]; then
        echo "${mid} MUT-CRASH  ${desc}  (rc=${rc})"
        CRASHED=$((CRASHED + 1))
    elif [ $rc -ge 128 ]; then
        echo "${mid} CAUGHT  ${desc}  [crashed AFTER reporting ${n_fail} failures -- the defect was DETECTED]"
        CAUGHT=$((CAUGHT + 1))
    else
        echo "${mid} CAUGHT  ${desc}  [${n_fail} failures]"
        echo "      ${first}"
        CAUGHT=$((CAUGHT + 1))
    fi
done <<< "${MUTATIONS}"

restore

echo ""
echo "=== SUMMARY: ${CAUGHT} caught, ${SURVIVED} survived, ${BROKEN} broken, ${CRASHED} crashed, ${NOOP} noop, ${ANCHOR_BAD} anchor ==="
if [ ${SURVIVED} -gt 0 ]; then
    echo "SURVIVORS ARE HOLES IN THE SUITE:"
    exit 1
fi
if [ ${BROKEN} -gt 0 ] || [ ${CRASHED} -gt 0 ] || [ ${NOOP} -gt 0 ] || [ ${ANCHOR_BAD} -gt 0 ]; then
    echo "HARNESS DEFECTS OR UNVERDICTED MUTATIONS (not passes):"
    exit 1
fi
exit 0
