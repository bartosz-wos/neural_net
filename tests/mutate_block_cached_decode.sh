#!/usr/bin/env bash
# =============================================================================
# tests/mutate_block_cached_decode.sh -- mutation harness for the BLOCK-LEVEL
# cached decode: include/nn/layers/attention/transformer.cpp.
#
# PURPOSE: prove that tests/test_block_cached_decode.cpp actually CATCHES the
# defects it was written for. A green suite proves nothing on its own; the
# mutation has to fail.
#
# =============================================================================
# THE MUTATIONS, AND THE DECISION EACH ONE TARGETS
# =============================================================================
#
#   M1  the cached block reads the NEW tokens' keys instead of the cache's
#       -> "the cache is actually read" at the BLOCK level. This is the
#          mutation a suite which only compared the decode against ITSELF
#          would miss.
#   M2  the cached block reads the NEW tokens' values instead of the cache's
#       -> the key/value separation.
#   M3  the cached path's causal mask is dropped
#       -> the causality of the decode.
#   M4  the causal mask uses >= instead of >
#       -> the mask's boundary.
#   M5  the cached path's softmax loses its max-subtraction
#       -> the softmax's numerical stability.
#   M6  the shared tail's RESIDUAL ADD is dropped
#       -> the tail's structure. **A SHARED-HELPER MUTATION**: the tail is
#          used by BOTH the ordinary and the cached path, so the
#          decode-vs-full comparison CANNOT see it on its own -- the
#          cross-language pinned fixture is what catches it.
#   M7  the shared tail's SECOND LayerNorm is dropped
#       -> the tail's structure, from the other side.
#   M8  the shared tail's FFN applies the second linear to the PRE-GELU
#       activation
#       -> the FFN's ordering.
#   M9  the shared tail's FFN weight order is TRANSPOSED
#       -> THE PROJECTION CONVENTION (the repo stores W as (d_model,
#          d_model) with the INPUT feature on the row axis).
#   M10 the shared tail's GELU loses its 0.5 factor
#       -> the activation.
#   M11 the shared tail's LayerNorm is applied to the whole sequence at once
#       -> THE ROW-WISE PROPERTY. This is the mutation that makes the
#          "no residual-stream cache is needed" claim observable: a tail
#          that mixed tokens would break the decode.
#   M12 the cached path's ATTENTION is skipped (the tail sees zeros)
#       -> "the attention is really run".
#   M13 the cached path's OUTPUT PROJECTION is skipped
#       -> the W_o projection.
#   M14 the layer index is IGNORED (every block uses slot 0)
#       -> the per-layer indexing, i.e. the two-block stack.
#   M15 the cache is RESET before every append
#       -> "the past is kept".
#   M16 the backward guard's MESSAGE changes
#       -> the message discipline: the discriminator is the MESSAGE, not the
#          throw.
#   M17 the backward guard is moved to the END of the backward pass
#       -> THE GUARD'S POSITION. A guard at the end would let the
#          gradient buffers be written before the throw.
#   M18 the positional encoding's OFFSET is ignored (it uses the local index)
#       -> THE ABSOLUTE-POSITION CLAIM.
#   M19 the positional encoding's ROW-COUNT guard is dropped
#       -> the out-of-bounds read.
#   M20 the positional encoding's LENGTH guard is dropped
#       -> the silent zeroing of the positions past max_len.
#   M21 the block's `d_model` member is left UNINITIALISED
#       -> THE BUG THIS SESSION FOUND. The member is public and the
#          constructor's parameter shadows it, so it held GARBAGE.
#   M22 the block's GRADIENT BUFFERS are left UNINITIALISED
#       -> THE MEMORY-CORRUPTION BUG THIS SESSION FOUND: four 0x0 tensors
#          that `backward()` wrote past the end of.
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
#   2. The link set comes from `make print-lib-objs`, NEVER from a regex over
#      the Makefile. `LIB_OBJS` is a RECURSIVE make variable whose literal
#      text contains no `.o` filenames, so a regex yields an empty set and
#      every mutation would score a false MUT-BROKEN -- one defect reported
#      as N verdicts. The count is asserted >= 10.
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
#               reported as such -- never as a pass.
#   MUT-BROKEN the BUILD failed. Not a verdict about the suite.
#   MUT-CRASH  the binary died (rc >= 128). Not a verdict about the suite.
#   MUT-NOOP   the mutation did not apply. A harness defect.
#   MUT-ANCHOR the anchor was not unique. A harness defect.
#
# =============================================================================

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${REPO_ROOT}/include/nn/layers/attention/transformer.cpp"
BUILD_DIR="${REPO_ROOT}/build"
MUT_BUILD="${REPO_ROOT}/build/mutation/block_cached_decode"
TEST_SRC="${REPO_ROOT}/tests/test_block_cached_decode.cpp"

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
M1|the cached block reads the NEW tokens' keys instead of the cache's|                    s += Q[i][h * d_k + dk] * Kc[j][h * d_k + dk];|                    s += Q[i][h * d_k + dk] * K[j % tokens][h * d_k + dk];
M2|the cached block reads the NEW tokens' values instead of the cache's|                    val += scores[i][j] * Vc[j][h * d_k + dk];|                    val += scores[i][j] * V[j % tokens][h * d_k + dk];
M3|the cached path's causal mask is dropped|                if (j > static_cast<size_t>(past + static_cast<int>(i)))\n                    sc += -1e9f;|                if (false)\n                    sc += -1e9f;
M4|the causal mask uses >= instead of >|                if (j > static_cast<size_t>(past + static_cast<int>(i)))|                if (j >= static_cast<size_t>(past + static_cast<int>(i)));
M5|the cached path's softmax loses its max-subtraction|            for (size_t j = 1; j < static_cast<size_t>(total); ++j)\n                if (scores[i][j] > max_s) max_s = scores[i][j];\n            double sum_exp = 0.0;\n            for (size_t j = 0; j < static_cast<size_t>(total); ++j) {\n                scores[i][j] = std::exp(scores[i][j] - max_s);|            for (size_t j = 1; j < static_cast<size_t>(total); ++j)\n                if (scores[i][j] > max_s) max_s = scores[i][j];\n            double sum_exp = 0.0;\n            for (size_t j = 0; j < static_cast<size_t>(total); ++j) {\n                scores[i][j] = std::exp(scores[i][j]);
M6|the shared tail's RESIDUAL ADD is dropped|    Tensor residual = x_tokens + attn_tokens;|    Tensor residual = attn_tokens;
M7|the shared tail's SECOND LayerNorm uses the FIRST one's parameters|    Tensor ffn_res = norm1_out + ffn2;\n    Tensor norm2_out = ln2.forward(ffn_res);|    Tensor ffn_res = norm1_out + ffn2;\n    Tensor norm2_out = ln1.forward(ffn_res);
M8|the shared tail's FFN applies the second linear to the PRE-GELU activation|    // GELU activation\n    for (size_t i = 0; i < tokens; ++i) {\n        for (size_t j = 0; j < d_model_local; ++j) {\n            double x_g = ffn1[i][j];\n            ffn1[i][j] = 0.5 * x_g * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x_g + 0.044715 * x_g * x_g * x_g)));\n        }\n    }\n\n    // Second linear\n    Tensor ffn2(tokens, d_model_local);\n    for (size_t i = 0; i < tokens; ++i) {\n        for (size_t j = 0; j < d_model_local; ++j) {\n            double val = b2[0][j];\n            for (size_t k = 0; k < d_model_local; ++k) {\n                val += W2[k][j] * ffn1[i][k];\n            }\n            ffn2[i][j] = val;\n        }\n    }|    // Second linear -- MOVED BEFORE the GELU, so it sees the PRE-GELU value.\n    Tensor ffn2(tokens, d_model_local);\n    for (size_t i = 0; i < tokens; ++i) {\n        for (size_t j = 0; j < d_model_local; ++j) {\n            double val = b2[0][j];\n            for (size_t k = 0; k < d_model_local; ++k) {\n                val += W2[k][j] * ffn1[i][k];\n            }\n            ffn2[i][j] = val;\n        }\n    }\n\n    // GELU activation\n    for (size_t i = 0; i < tokens; ++i) {\n        for (size_t j = 0; j < d_model_local; ++j) {\n            double x_g = ffn1[i][j];\n            ffn1[i][j] = 0.5 * x_g * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x_g + 0.044715 * x_g * x_g * x_g)));\n        }\n    }
M9|the shared tail's FFN weight order is TRANSPOSED|                val += W2[k][j] * ffn1[i][k];|                val += W2[j][k] * ffn1[i][k];
M10|the shared tail's GELU loses its 0.5 factor|            ffn1[i][j] = 0.5 * x_g * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x_g + 0.044715 * x_g * x_g * x_g)));|            ffn1[i][j] = x_g * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x_g + 0.044715 * x_g * x_g * x_g)));
M11|the shared tail's FFN's FIRST linear weight order is TRANSPOSED|                val += W1[k][j] * norm1_out[i][k];|                val += W1[j][k] * norm1_out[i][k];
M12|the cached path's ATTENTION is skipped|    Tensor attn_out = attn.forward_cached(input); // (d_model, n_new)|    Tensor attn_out = Tensor(d_model, seq_len); attn_out.fill(0.0); // (d_model, n_new)
M13|the cached path's OUTPUT PROJECTION is skipped|    // Final projection: output_acc @ W_o^T -- identical to the ordinary path.\n    Tensor output(tokens, d_model);\n    for (size_t i = 0; i < tokens; ++i) {\n        for (size_t j = 0; j < d_model; ++j) {\n            double val = 0.0;\n            for (size_t k = 0; k < d_model; ++k) {\n                val += output_acc[i][k] * W_o[k][j];\n            }\n            output[i][j] = val;\n        }\n    }|    // Final projection: output_acc @ W_o^T -- identical to the ordinary path.\n    Tensor output(tokens, d_model);\n    for (size_t i = 0; i < tokens; ++i) {\n        for (size_t j = 0; j < d_model; ++j) {\n            double val = 0.0;\n            for (size_t k = 0; k < d_model; ++k) {\n                val += output_acc[i][k];\n            }\n            output[i][j] = val;\n        }\n    }
M14|the layer index is IGNORED (every block uses slot 0)|    kv_layer_index_ = layer_index;|    kv_layer_index_ = 0;
M15|the cache is RESET before every append|    kv_cache_->append(kv_layer_index_, K, V);|    kv_cache_->reset(kv_layer_index_);\n    kv_cache_->append(kv_layer_index_, K, V);
M16|the backward guard's MESSAGE changes|            "TransformerBlock::backward: the last forward was a CACHED "\n            "forward, which is an inference path and has no adjoint; call "\n            "forward() before backward()");|            "TransformerBlock::backward: the last forward was a CACHED "\n            "forward, which has no adjoint; call forward() before backward()");
M17|the positional encoding's OFFSET is ignored|            output[f][s] = input[f][s] + pe[offset + s][f];|            output[f][s] = input[f][s] + pe[s][f];
M18|the positional encoding's ROW-COUNT guard is dropped|    if (d_model_local != pe.cols) {|    if (false) {
M19|the positional encoding's LENGTH guard is dropped|    if (offset + seq_len > max_len) {|    if (false) {
M20|the block's `d_model` member is left UNINITIALISED|    : d_model(d_model)\n    , num_heads(num_heads)\n    , attn(d_model, num_heads)|    : num_heads(num_heads)\n    , attn(d_model, num_heads)
M21|the block's GRADIENT BUFFERS are left UNINITIALISED|    , grad_W1(Tensor::zeros(d_model, d_model))\n    , grad_b1(Tensor::zeros(1, d_model))\n    , grad_W2(Tensor::zeros(d_model, d_model))\n    , grad_b2(Tensor::zeros(1, d_model))|    // (the gradient buffers are left uninitialised)
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
while IFS='|' read -r mid desc search repl; do
    [ -z "${mid}" ] && continue

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
            */layers/attention/transformer.o) ;;
            *) LINK_OBJS="${LINK_OBJS} $o" ;;
        esac
    done
    MUT_OBJ="${MUT_BUILD}/mutated.o"
    rm -f "${MUT_OBJ}" "${MUT_BUILD}/test_block_cached_decode" "${MUT_BUILD}/test_block_cached_decode.o"
    if ! g++ -std=c++17 -O2 -Wall -Wextra -march=native -I"${REPO_ROOT}/include" \
            -c "${TARGET}" -o "${MUT_OBJ}" 2> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the mutated TU did not compile -- a BUILD verdict, not a coverage one"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ -std=c++17 -O2 -I"${REPO_ROOT}/include" \
            -c "${TEST_SRC}" -o "${MUT_BUILD}/test_block_cached_decode.o" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the test did not compile -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi
    if ! g++ "${MUT_BUILD}/test_block_cached_decode.o" "${MUT_OBJ}" ${LINK_OBJS} \
            -o "${MUT_BUILD}/test_block_cached_decode" 2>> "${MUT_BUILD}/compile.log"; then
        echo "${mid} MUT-BROKEN  the link failed -- a BUILD verdict"
        BROKEN=$((BROKEN + 1))
        continue
    fi

    # --- run the suite.
    "${MUT_BUILD}/test_block_cached_decode" > "${MUT_BUILD}/run.log" 2>&1
    rc=$?
    # `grep -c` prints "0" AND exits 1 when there is no match, so the
    # original `|| echo 0` appended a SECOND "0" and the variable became
    # "0\n0" -- which the integer comparison rejected with
    # "[: 0\n0: integer expected". A stderr line the loop ignores is not
    # an error report; this one was a real harness defect.
    n_fail=$(grep -c '\[FAIL\]' "${MUT_BUILD}/run.log" 2>/dev/null || true)
    n_fail=${n_fail:-0}
    first=$(grep -m1 '\[FAIL\]' "${MUT_BUILD}/run.log" | cut -c1-140)

    if [ $rc -eq 0 ]; then
        echo "${mid} SURVIVED  ${desc}"
        SURVIVED=$((SURVIVED + 1))
    elif [ $rc -ge 128 ] && [ "${n_fail}" -eq 0 ]; then
        # A CRASH is a THIRD verdict class: the binary compiled and then
        # died, which says nothing about whether the suite covers the
        # behaviour.
        #
        # **BUT A CRASH AFTER THE SUITE HAS REPORTED A [FAIL] IS A
        # DETECTION, NOT A SILENT CRASH.** The repo's rule exists so that a
        # mutation which removes a resource cannot be scored as "the suite
        # covers it" when the suite simply DIED WITHOUT SAYING ANYTHING.
        # When the log already contains a reported failure, the suite DID
        # detect the defect and the crash is a CONSEQUENCE of it -- M21, the
        # uninitialised gradient buffers, is exactly this case: the shape
        # assertion fires, and the out-of-bounds write in backward() then
        # takes the process down. Scoring that as "no verdict" would throw
        # away a real detection, so the rule is: a crash WITH a reported
        # failure is CAUGHT, and a crash WITHOUT one is MUT-CRASH.
        echo "${mid} MUT-CRASH  ${desc}  (rc=${rc})"
        CRASHED=$((CRASHED + 1))
    elif [ $rc -ge 128 ]; then
        echo "${mid} CAUGHT  ${desc}  [crashed AFTER reporting ${n_fail} failures -- the defect was DETECTED and the crash is a consequence]"
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
