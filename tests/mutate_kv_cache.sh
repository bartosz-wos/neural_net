#!/usr/bin/env bash
# =============================================================================
# tests/mutate_kv_cache.sh -- mutation harness for the inference-time KV
# cache: include/nn/utils/kv_cache.cpp and
# include/nn/layers/attention/transformer.cpp.
#
# PURPOSE: prove that tests/test_kv_cache.cpp actually CATCHES the defects
# it was written for. A green suite proves nothing on its own; the mutation
# has to fail.
#
# =============================================================================
# THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
# =============================================================================
#
#   M1  the cached path's causal mask is dropped
#       -> the causality of the decode. A decode that attends to the future
#          is the classic cache bug.
#   M2  the cached path reads the NEW tokens' keys instead of the cache's
#       -> "the cache is actually read". This is the mutation that a
#          suite which only compares the decode against ITSELF would miss.
#   M3  the cached path reads the NEW tokens' values instead of the cache's
#       -> the key/value separation.
#   M4  the causal mask uses >= instead of >
#       -> the mask's boundary. The diagonal is the token's own key.
#   M5  the cached path's attention scale loses the + 1e-9
#       -> the scale convention.
#   M6  the cached path's softmax loses its max-subtraction
#       -> the softmax's numerical stability.
#   M7  the cache is RESET before every append
#       -> "the past is kept". A cache that forgets its past is the
#          second classic cache bug.
#   M8  the layer index is IGNORED (every layer uses slot 0)
#       -> the per-layer indexing, i.e. the two-layer stack.
#   M9  the shared Q/K/V projection is TRANSPOSED
#       -> the repo's projection convention -- caught by the CROSS-LANGUAGE
#          pinned fixture, and NOT by the decode-vs-full comparison (both
#          paths share the helper, so a shared mutation cancels).
#   M10 the cached path skips the output projection
#       -> the W_o projection.
#   M11 the backward guard's MESSAGE changes
#       -> the message discipline: the discriminator is the message, not
#          the throw. The guard ITSELF is not removed, because removing it
#          makes the backward read an empty score buffer, which CRASHES --
#          and a crash is a third verdict class, not a coverage one.
#   M12 the cache's length grows by ONE row whatever is appended
#       -> the bookkeeping. **THE FIRST DRAFT OF THIS MUTATION DID NOT
#          INCREMENT THE LENGTH AT ALL**, which leaves the cached path
#          reading an EMPTY score buffer and SEGFAULTS -- a crash, which is
#          a third verdict class and says nothing about the suite. It was
#          rewritten to grow the length by one row instead, which changes
#          the observable without emptying a buffer the code indexes.
#   M13 the appended keys land in the VALUE buffer
#       -> the key/value separation, in the cache.
#   M14 the second append OVERWRITES the first
#       -> the append offset.
#   M15 the cache's WIDTH guard is dropped
#       -> the dimension guard. With the guard gone the append SUCCEEDS
#          with a wrong width, so the message assertion is what catches it.
#   M16 the cache's ROW-COUNT guard is dropped
#       -> the row-count guard.
#   M17 the cache's ZERO-ROWS guard is dropped
#       -> the zero-row guard.
#   M18 total_tokens counts the LAYERS, not the tokens
#       -> the accounting.
#   M19 memory_bytes counts ONE buffer per layer
#       -> the accounting, from the other side.
#   M20 reset does not clear the layer
#       -> the reset.
#
# =============================================================================
# THE HARNESS RULES THIS REPO HAS ALREADY LEARNED
# =============================================================================
#
#   1. `restore()` must NEVER delete its backup. It is called once PER
#      MUTATION and once on EXIT, so a self-cleaning restore works exactly
#      once and every later restore is a silent no-op -- leaving a MUTANT
#      IN THE TREE. Cleanup happens exactly once, on EXIT, and restore() is
#      ATOMIC (`cp tmp && mv`), because a plain `cp` truncates the
#      destination first, so a concurrent reader can splice two generations
#      together.
#   2. The link set comes from `make print-lib-objs`, NEVER from
#      regex-parsing the Makefile. `LIB_OBJS` is a RECURSIVE make variable
#      whose literal text contains no `.o` filenames, so a regex yields an
#      empty set and every mutation would score a false MUT-BROKEN -- one
#      defect reported as N verdicts. The count is asserted >= 10.
#   3. THE MUTATED OBJECT MUST NOT BE IN ITS OWN LINK SET. It legitimately IS
#      in the library's list (it is the TU being mutated), so the mutated TU
#      is compiled SEPARATELY and swapped in for the library's copy.
#   4. `MUT-BROKEN` means "the build failed". It is a SEPARATE CLASS and is
#      never counted as a pass. A mutation reported BROKEN is re-run in
#      isolation before it is believed.
#   5. `rc >= 128` is a CRASH, which is a THIRD verdict class: the binary
#      compiled and then died, which says NOTHING about whether the suite
#      covers the behaviour. It is not reported as CAUGHT.
#   6. `rm -f` the intermediates before every build. A stale `.o` from the
#      previous mutation produces a phantom failure.
#   7. NEVER run a build while this harness is running: two harnesses that
#      mutate the same source file race on `build/*.o`.
#   8. The ANCHOR must be UNIQUE. A line that appears twice makes the edit
#      a coin flip, and a coin flip is not a verdict. Each anchor's count
#      is asserted to be exactly 1 before the edit.
#   9. The edits are done with a PYTHON script, not perl/sed: this repo has
#      been bitten by perl's s/// delimiter (the anchors are full of "/")
#      and by sed interpreting a newline inside a pattern. Python's
#      str.count / str.replace has no delimiter concept at all.
#
# =============================================================================
# THE VERDICT CLASSES
# =============================================================================
#
#   CAUGHT     the suite FAILED. This is the only good outcome.
#   SURVIVED   the suite PASSED. This is a hole in the SUITE, and it is
#              reported as such -- never as a pass.
#   MUT-BROKEN the BUILD failed. Not a verdict about the suite.
#   MUT-CRASH  the binary died (rc >= 128). Not a verdict about the suite.
#   MUT-NOOP   the mutation did not apply. A harness defect.
#   MUT-ANCHOR the anchor was not unique. A harness defect.
#
# =============================================================================

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_SRC="${REPO_ROOT}/include/nn/utils/kv_cache.cpp"
MHA_SRC="${REPO_ROOT}/include/nn/layers/attention/transformer.cpp"
BUILD_DIR="${REPO_ROOT}/build"
MUT_BUILD="${REPO_ROOT}/build/mutation/kv_cache"
TEST_SRC="${REPO_ROOT}/tests/test_kv_cache.cpp"

CAUGHT=0
SURVIVED=0
BROKEN=0
CRASHED=0
NOOP=0
ANCHOR_BAD=0

# --- the backups. restore() NEVER deletes them; cleanup does, on EXIT. ----
CACHE_BACKUP="$(mktemp)"
MHA_BACKUP="$(mktemp)"
cp "${CACHE_SRC}" "${CACHE_BACKUP}"
cp "${MHA_SRC}" "${MHA_BACKUP}"

restore() {
    # ATOMIC: cp to a temp then mv, so a concurrent reader never sees a
    # half-written source.
    cp "${CACHE_BACKUP}" "${CACHE_SRC}.tmp" && mv -f "${CACHE_SRC}.tmp" "${CACHE_SRC}"
    cp "${MHA_BACKUP}" "${MHA_SRC}.tmp" && mv -f "${MHA_SRC}.tmp" "${MHA_SRC}"
}

cleanup() {
    restore
    rm -f "${CACHE_BACKUP}" "${MHA_BACKUP}"
    rm -rf "${MUT_BUILD}"
}
trap cleanup EXIT INT TERM

# =============================================================================
# The mutation table.
#   id | which TU | description | python search | python replacement
# =============================================================================
read -r -d '' MUTATIONS <<'PYEOF'
M1|mha|the cached path's causal mask is dropped|                if (j > static_cast<size_t>(past + static_cast<int>(i)))\n                    sc += -1e9f;|                if (false)\n                    sc += -1e9f;
M2|mha|the cached path reads the NEW tokens' keys instead of the cache's|                    s += Q[i][h * d_k + dk] * Kc[j][h * d_k + dk];|                    s += Q[i][h * d_k + dk] * K[j % tokens][h * d_k + dk];
M3|mha|the cached path reads the NEW tokens' values instead of the cache's|                    val += scores[i][j] * Vc[j][h * d_k + dk];|                    val += scores[i][j] * V[j % tokens][h * d_k + dk];
M4|mha|the causal mask uses >= instead of >|                if (j > static_cast<size_t>(past + static_cast<int>(i)))|                if (j >= static_cast<size_t>(past + static_cast<int>(i)));
M5|mha|the cached path's attention scale loses the + 1e-9|                double sc = s / std::sqrt((double)d_k + 1e-9);|                double sc = s / std::sqrt((double)d_k);
M6|mha|the cached path's softmax loses its max-subtraction|            for (size_t j = 1; j < static_cast<size_t>(total); ++j)\n                if (scores[i][j] > max_s) max_s = scores[i][j];\n            double sum_exp = 0.0;\n            for (size_t j = 0; j < static_cast<size_t>(total); ++j) {\n                scores[i][j] = std::exp(scores[i][j] - max_s);|            for (size_t j = 1; j < static_cast<size_t>(total); ++j)\n                if (scores[i][j] > max_s) max_s = scores[i][j];\n            double sum_exp = 0.0;\n            for (size_t j = 0; j < static_cast<size_t>(total); ++j) {\n                scores[i][j] = std::exp(scores[i][j]);
M7|mha|the cache is RESET before every append|    kv_cache_->append(kv_layer_index_, K, V);|    kv_cache_->reset(kv_layer_index_);\n    kv_cache_->append(kv_layer_index_, K, V);
M8|mha|the layer index is IGNORED (every layer uses slot 0)|    kv_layer_index_ = layer_index;|    kv_layer_index_ = 0;
M9|mha|the shared Q/K/V projection is TRANSPOSED|                qv += x[i][k] * W_q[k][j];\n                kv += x[i][k] * W_k[k][j];\n                vv += x[i][k] * W_v[k][j];|                qv += x[i][k] * W_q[j][k];\n                kv += x[i][k] * W_k[j][k];\n                vv += x[i][k] * W_v[j][k];
M10|mha|the cached path skips the output projection|    // Final projection: output_acc @ W_o^T -- identical to the ordinary path.\n    Tensor output(tokens, d_model);\n    for (size_t i = 0; i < tokens; ++i) {\n        for (size_t j = 0; j < d_model; ++j) {\n            double val = 0.0;\n            for (size_t k = 0; k < d_model; ++k) {\n                val += output_acc[i][k] * W_o[k][j];|    // Final projection: output_acc @ W_o^T -- identical to the ordinary path.\n    Tensor output(tokens, d_model);\n    for (size_t i = 0; i < tokens; ++i) {\n        for (size_t j = 0; j < d_model; ++j) {\n            double val = 0.0;\n            for (size_t k = 0; k < d_model; ++k) {\n                val += output_acc[i][k];
M11|mha|the backward guard's MESSAGE changes|            "MultiHeadAttention::backward: the last forward was a CACHED "\n            "forward, which is an inference path and has no adjoint; call "\n            "forward() before backward()");|            "MultiHeadAttention::backward: the last forward was a CACHED "\n            "forward, which has no adjoint; call forward() before backward()");
M12|cache|the cache's length grows by ONE row whatever is appended|    l.len += static_cast<int>(k.rows);|    l.len += 1;
M13|cache|the appended keys land in the VALUE buffer|            l.keys(at + static_cast<int>(i), j) = k(i, j);\n            l.values(at + static_cast<int>(i), j) = v(i, j);|            l.values(at + static_cast<int>(i), j) = k(i, j);\n            l.values(at + static_cast<int>(i), j) = v(i, j);
M14|cache|the second append OVERWRITES the first|            l.keys(at + static_cast<int>(i), j) = k(i, j);\n            l.values(at + static_cast<int>(i), j) = v(i, j);|            l.keys(static_cast<int>(i), j) = k(i, j);\n            l.values(at + static_cast<int>(i), j) = v(i, j);
M15|cache|the cache's WIDTH guard is dropped|    if (static_cast<int>(k.cols) != l.width) {|    if (false) {
M16|cache|the cache's ROW-COUNT guard is dropped|    if (k.rows != v.rows) {|    if (false) {
M17|cache|the cache's ZERO-ROWS guard is dropped|    if (k.rows == 0) {|    if (false) {
M18|cache|total_tokens counts the LAYERS, not the tokens|        total += layers_[i].len;|        total += 1;
M19|cache|memory_bytes counts ONE buffer per layer|        bytes += 2LL * capacity_ * layers_[i].width * static_cast<long long>(sizeof(double));|        bytes += 1LL * capacity_ * layers_[i].width * static_cast<long long>(sizeof(double));
M20|cache|reset does not clear the layer|    layers_[layer].len = 0;\n}\n\nvoid KVCache::reset_all() {|    layers_[layer].len = 1;\n}\n\nvoid KVCache::reset_all() {
PYEOF

# =============================================================================
# The link set, from `make print-lib-objs` -- NEVER from a regex over the
# Makefile. A regex yields ZERO objects (the variable is recursive) and
# every mutation would then score a false MUT-BROKEN.
# =============================================================================
LIB_OBJS="$(cd "${REPO_ROOT}" && make -s print-lib-objs | tr '\n' ' ')"
LIB_COUNT="$(echo "${LIB_OBJS}" | wc -w)"
if [ "${LIB_COUNT}" -lt 10 ]; then
    echo "MUT-HARNESS-BROKEN: the link set has only ${LIB_COUNT} objects (must be >= 10)"
    echo "A regex over the Makefile would have produced an EMPTY set here, and"
    echo "every mutation would score a false MUT-BROKEN."
    exit 2
fi
echo "link set: ${LIB_COUNT} objects"

mkdir -p "${MUT_BUILD}"

# =============================================================================
# The main loop.
# =============================================================================
while IFS='|' read -r mid which desc search repl; do
    [ -z "${mid}" ] && continue

    if [ "${which}" = "cache" ]; then
        TARGET="${CACHE_SRC}"
        TARGET_OBJ="utils/kv_cache.o"
    else
        TARGET="${MHA_SRC}"
        TARGET_OBJ="layers/attention/transformer.o"
    fi

    # --- restore FIRST. Every iteration starts from the pristine sources.
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
    # NOT unique: the anchor is ambiguous, and an ambiguous edit is a coin
    # flip, not a verdict.
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
    if cmp -s "${TARGET}" "$(if [ "${which}" = "cache" ]; then echo "${CACHE_BACKUP}"; else echo "${MHA_BACKUP}"; fi)"; then
        echo "${mid} MUT-NOOP  the source is unchanged -- harness defect"
        NOOP=$((NOOP + 1))
        continue
    fi

    # --- build. The mutated TU is compiled separately and swapped in; the
    #     library's copy of it is dropped from the link set.
    LINK_OBJS=""
    for o in ${LIB_OBJS}; do
        case "$o" in
            */${TARGET_OBJ}) ;;
            *) LINK_OBJS="${LINK_OBJS} $o" ;;
        esac
    done
    MUT_OBJ="${MUT_BUILD}/mutated.o"
    rm -f "${MUT_OBJ}" "${MUT_BUILD}/test_kv_cache" "${MUT_BUILD}/test_kv_cache.o"
    if ! g++ -std=c++17 -O2 -Wall -Wextra -march=native -I"${REPO_ROOT}/include" \
            -c "${TARGET}" -o "${MUT_OBJ}" 2> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the mutated TU did not compile -- a BUILD verdict, not a coverage one"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ -std=c++17 -O2 -I"${REPO_ROOT}/include" \
            -c "${TEST_SRC}" -o "${MUT_BUILD}/test_kv_cache.o" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the test did not compile -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ "${MUT_BUILD}/test_kv_cache.o" "${MUT_OBJ}" ${LINK_OBJS} \
            -o "${MUT_BUILD}/test_kv_cache" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the link failed -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi

    # --- run the suite.
    "${MUT_BUILD}/test_kv_cache" > "${MUT_BUILD}/run.log" 2>&1
    rc=$?
    n_fail=$(grep -c '\[FAIL\]' "${MUT_BUILD}/run.log" 2>/dev/null || echo 0)
    first=$(grep -m1 '\[FAIL\]' "${MUT_BUILD}/run.log" | cut -c1-140)

    if [ $rc -eq 0 ]; then
        echo "${mid} SURVIVED  ${desc}"
        SURVIVED=$((SURVIVED + 1))
    elif [ $rc -ge 128 ]; then
        # A CRASH is a THIRD verdict class: the binary compiled and then
        # died, which says nothing about whether the suite covers the
        # behaviour.
        echo "${mid} MUT-CRASH  ${desc}  (rc=${rc})"
        CRASHED=$((CRASHED + 1))
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
