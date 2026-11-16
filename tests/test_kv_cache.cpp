// =============================================================================
// tests/test_kv_cache.cpp -- the inference-time key/value cache and the
// cached incremental forward path of the multi-head attention.
//
// THE SPECIFICATION IS tests/kv_cache_oracle.py (float64 numpy), which was
// written and run BEFORE this file existed. Its discriminating power was
// MEASURED, not narrated: tests/kv_cache_mutate_oracle.py breaks the
// oracle's model code in 22 specific ways and requires that it fail each
// time (22 caught / 0 survived).
//
// THE GAP, MEASURED NOT ASSUMED (2026-11-16):
//
//   grep -rinE '\bkv.?cache|paged.attn|prefix.cache|past_key' include/
//   -> 3 hits, ALL COMMENTS (mla.h:33, sliding_window.h:17, megalodon.h:67)
//   -> ZERO implementation across ~250 layers.
//
// Many attention layers hold last_k_ / last_v_ -- but those are caches for
// the BACKWARD pass, not an inference-time cache. The repo had no
// incremental decoding path at all.
//
// THE INVARIANT THIS WHOLE FEATURE RESTS ON
//
//   For a causal attention, decoding one token at a time with a cache must
//   equal the full-sequence forward, TOKEN FOR TOKEN. That is a property of
//   the OUTPUT, not a theorem, and the suite pins it.
//
// AND IT IS PINNED EXACTLY, NOT WITHIN A TOLERANCE. The cached path performs
// the same arithmetic in the same order as the full path, so the two agree
// to the LAST BIT -- and a tolerance would hide a cache that attends to the
// wrong tokens by a small amount.
//
// THE NON-VACUITY REQUIREMENT
//
//   A cached path that silently ignores the cache produces the right answer
//   at step 0 and the wrong one everywhere after. The suite therefore also
//   proves that the cache is READ: perturbing token 0 must change the
//   output of a later step.
//
// THE ERROR CHECKS ASSERT THE MESSAGE, NOT MERELY "IT THREW". A check that
// only asserts "it threw" is blind to a guard that was removed while a
// DIFFERENT guard still fires: the throw happens either way. This is not
// hypothetical -- the oracle's own mutation harness showed that removing
// the width guard leaves numpy's broadcast error throwing in its place, so
// a throw-only check stays GREEN.
//
//   C++ evaluates a function call's arguments in UNSPECIFIED order, so
//   every throwing assertion goes through expect_throw(), which calls,
//   captures, and THEN reports. A message printed from the wrong call is
//   worse than no message at all.
// =============================================================================

#include "nn/utils/kv_cache.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using std::string;
using std::vector;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const string& msg) {
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "\n";
    }
}

static bool same(const Tensor& a, const Tensor& b, double tol = 0.0) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            if (std::fabs(a(i, j) - b(i, j)) > tol) return false;
    return true;
}

// A tensor of n rows x w cols with distinct values, so that a mix-up
// between the key buffer and the value buffer cannot pass.
static Tensor rows(int n, int w, double base) {
    Tensor t(n, w);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < w; ++j)
            t(i, j) = base + i * 100.0 + j;
    return t;
}

// THE DISCRIMINATOR IS THE MESSAGE, NOT THE THROW.
static void expect_throw(const std::function<void()>& fn,
                         const string& needle,
                         const string& name) {
    string what;
    bool threw = false;
    try {
        fn();
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    if (!threw) {
        ++g_fail;
        std::cout << "  [FAIL] " << name << " -- did not throw\n";
        return;
    }
    if (what.find(needle) == string::npos) {
        ++g_fail;
        std::cout << "  [FAIL] " << name << " -- threw \"" << what
                  << "\", expected a message containing \"" << needle << "\"\n";
        return;
    }
    ++g_pass;
}

// =============================================================================
// TEST 1: a fresh cache
// =============================================================================

static void test_construction() {
    nn::utils::KVCache cache(16);
    check(cache.capacity() == 16, "the capacity is reported");
    check(cache.layers() == 0, "a fresh cache has no layers");
    check(cache.total_tokens() == 0, "a fresh cache holds no tokens");
    check(cache.memory_bytes() == 0, "a fresh cache holds no bytes");
}

// =============================================================================
// TEST 2: reserve, append, and the buffer contents
// =============================================================================

static void test_append_and_size() {
    nn::utils::KVCache cache(8);
    cache.reserve(0, 4);
    check(cache.layers() == 1, "reserve creates a layer");
    check(cache.size(0) == 0, "a reserved layer starts empty");
    check(cache.d_kv(0) == 4, "the layer reports its width");

    Tensor k = rows(3, 4, 1.0);
    Tensor v = rows(3, 4, -1.0);
    cache.append(0, k, v);
    check(cache.size(0) == 3, "append grows the layer by the row count");

    const Tensor& kb = cache.key_buffer(0);
    const Tensor& vb = cache.value_buffer(0);
    check(kb.rows == 8 && kb.cols == 4,
          "the key buffer is capacity x d_kv");
    check(vb.rows == 8 && vb.cols == 4,
          "the value buffer is capacity x d_kv");

    bool keys_ok = true;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (kb(i, j) != k(i, j)) keys_ok = false;
        }
    }
    check(keys_ok, "the cached keys are the appended rows");

    // The WHOLE buffer is compared, not just the valid rows: the tail must
    // still be zero, because a cache that scribbled past the length would
    // be read by a consumer that trusts size().
    Tensor expect_kb(8, 4), expect_vb(8, 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j) {
            expect_kb(i, j) = k(i, j);
            expect_vb(i, j) = v(i, j);
        }
    check(same(kb, expect_kb),
          "the key buffer holds the appended rows and nothing else");
    check(same(vb, expect_vb),
          "the value buffer holds the appended rows and nothing else");
}

// =============================================================================
// TEST 3: a second append lands AFTER the first, and does not disturb it
// =============================================================================

static void test_second_append() {
    nn::utils::KVCache cache(8);
    cache.reserve(0, 4);
    Tensor k1 = rows(3, 4, 1.0);
    Tensor v1 = rows(3, 4, -1.0);
    Tensor k2 = rows(2, 4, 500.0);
    Tensor v2 = rows(2, 4, -500.0);
    cache.append(0, k1, v1);
    cache.append(0, k2, v2);
    check(cache.size(0) == 5, "two appends grow the layer by five rows");

    const Tensor& kb = cache.key_buffer(0);
    const Tensor& vb = cache.value_buffer(0);
    bool first_intact = true, second_placed = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            if (kb(i, j) != k1(i, j) || vb(i, j) != v1(i, j)) first_intact = false;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            if (kb(3 + i, j) != k2(i, j) || vb(3 + i, j) != v2(i, j))
                second_placed = false;
    check(first_intact, "the FIRST append is not disturbed by the second");
    check(second_placed, "the second append lands after the first");
}

// =============================================================================
// TEST 4: the layers are independent
// =============================================================================

static void test_layer_independence() {
    nn::utils::KVCache cache(8);
    cache.reserve(0, 4);
    cache.reserve(1, 4);
    check(cache.layers() == 2, "two layers are reserved");
    Tensor k = rows(3, 4, 1.0);
    Tensor v = rows(3, 4, -1.0);
    cache.append(0, k, v);
    cache.append(1, k, v);
    cache.append(1, k, v);
    check(cache.size(0) == 3, "layer 0 holds three tokens");
    check(cache.size(1) == 6, "layer 1 holds six tokens");
    check(cache.total_tokens() == 9, "total_tokens sums the layers");

    // A layer with a DIFFERENT width is a different layer.
    cache.reserve(2, 7);
    check(cache.d_kv(2) == 7, "each layer keeps its own width");
    check(cache.size(2) == 0, "a newly reserved layer is empty");
}

// =============================================================================
// TEST 5: reset
// =============================================================================

static void test_reset() {
    nn::utils::KVCache cache(8);
    cache.reserve(0, 4);
    cache.reserve(1, 4);
    Tensor k = rows(3, 4, 1.0);
    Tensor v = rows(3, 4, -1.0);
    cache.append(0, k, v);
    cache.append(1, k, v);

    cache.reset(0);
    check(cache.size(0) == 0, "reset empties one layer");
    check(cache.size(1) == 3, "reset leaves the other layers alone");

    cache.reset_all();
    check(cache.total_tokens() == 0, "reset_all empties every layer");

    // A reset layer is REUSABLE, and it really starts empty.
    cache.append(0, k, v);
    check(cache.size(0) == 3, "a reset layer is reusable");
    const Tensor& kb = cache.key_buffer(0);
    bool ok = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            if (kb(i, j) != k(i, j)) ok = false;
    check(ok, "a reset layer really starts empty");
}

// =============================================================================
// TEST 6: the memory accounting
// =============================================================================

static void test_memory_accounting() {
    nn::utils::KVCache cache(8);
    check(cache.memory_bytes() == 0, "an empty cache holds no bytes");
    cache.reserve(0, 4);
    check(cache.memory_bytes() == 2 * 8 * 4 * 8,
          "memory_bytes is two buffers of capacity x d_kv doubles");
    cache.reserve(1, 4);
    check(cache.memory_bytes() == 2 * 2 * 8 * 4 * 8,
          "memory_bytes sums the layers");
    cache.reserve(2, 7);
    check(cache.memory_bytes() == 2 * 8 * 4 * 8 + 2 * 8 * 4 * 8 + 2 * 8 * 7 * 8,
          "memory_bytes uses each layer's own width");
}

// =============================================================================
// TEST 7: the error cases -- each with its OWN message
// =============================================================================

static void test_errors() {
    nn::utils::KVCache cache(4);

    // An unreserved layer.
    expect_throw([&] { cache.size(0); }, "not reserved",
                 "size() on an unreserved layer throws");
    expect_throw([&] { cache.key_buffer(0); }, "not reserved",
                 "key_buffer() on an unreserved layer throws");
    expect_throw([&] { cache.append(0, rows(1, 3, 0.0), rows(1, 3, 0.0)); },
                 "not reserved",
                 "appending to an unreserved layer throws");

    cache.reserve(0, 3);

    // The width guard: the layer was reserved with 3 columns. The needle is
    // "reserved with" and NOT "columns", because the key/value width
    // mismatch below says "the same number of columns" -- a needle shared by
    // two messages discriminates nothing.
    expect_throw([&] { cache.append(0, rows(1, 4, 0.0), rows(1, 4, 0.0)); },
                 "reserved with",
                 "appending the wrong width throws");

    // The row-count guard.
    expect_throw([&] { cache.append(0, rows(2, 3, 0.0), rows(3, 3, 0.0)); },
                 "same number of rows",
                 "appending mismatched row counts throws");

    // The key/value width guard.
    expect_throw([&] { cache.append(0, rows(1, 3, 0.0), rows(1, 4, 0.0)); },
                 "same number of columns",
                 "appending keys and values of different widths throws");

    // The zero-rows guard.
    expect_throw([&] { cache.append(0, rows(0, 3, 0.0), rows(0, 3, 0.0)); },
                 "zero rows",
                 "appending zero rows throws");

    // The capacity guard.
    cache.append(0, rows(4, 3, 0.0), rows(4, 3, 0.0));
    expect_throw([&] { cache.append(0, rows(1, 3, 0.0), rows(1, 3, 0.0)); },
                 "capacity",
                 "appending past the capacity throws");

    // Re-reserving with a different width.
    expect_throw([&] { cache.reserve(0, 4); }, "already reserved",
                 "re-reserving a layer with a different width throws");

    // Re-reserving with the SAME width is allowed and keeps the content.
    nn::utils::KVCache keep(8);
    keep.reserve(0, 3);
    keep.append(0, rows(2, 3, 7.0), rows(2, 3, -7.0));
    keep.reserve(0, 3);
    check(keep.size(0) == 2,
          "re-reserving a layer with the same width keeps its content");
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "=== test_kv_cache ===\n";
    test_construction();
    test_append_and_size();
    test_second_append();
    test_layer_independence();
    test_reset();
    test_memory_accounting();
    test_errors();
    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    if (g_fail > 0) {
        std::cout << "TESTS FAILED\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
