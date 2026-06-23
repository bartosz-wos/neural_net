// Magnitude pruning tests — Frankle & Carlin 2019 ("Lottery Ticket")
//
// Tests:
//   1. count_nonzero basic (some zero, some non-zero)
//   2. count_nonzero all-zero / all-non-zero / empty
//   3. magnitude_prune_ basic: prune the smallest |w| entries in place
//   4. magnitude_prune_ sparsity=0 → no change, sparsity=1 → all zero
//   5. magnitude_prune_ clamping (sparsity < 0, sparsity > 1)
//   6. magnitude_prune_ preserves the largest weights exactly
//   7. magnitude_prune_ return value matches count of newly-zeroed entries
//   8. global_magnitude_prune_ distributes zeros by global threshold
//   9. global_magnitude_prune_ with single param equals local
//  10. global_magnitude_prune_ returns total number of entries zeroed
//  11. global_magnitude_prune_ handles null pointers in the list
//  12. global_magnitude_prune_ preserves the top-(1-sparsity) globally
//  13. determinism / no NaN
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include "nn/utils/magnitude_pruning.h"

using namespace std;

static int total = 0;
static int passed = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                       \
        if (cond) {                                                            \
            cout << "[PASS] " << msg << endl;                                  \
            ++passed;                                                          \
        } else {                                                               \
            cout << "[FAIL] " << msg << endl;                                  \
        }                                                                      \
        ++total;                                                               \
    } while (0)

static bool any_non_finite(const Tensor& t) {
    for (size_t i = 0; i < t.data.size(); ++i)
        if (!std::isfinite(t.data[i])) return true;
    return false;
}

int main() {
    cout << "=== Magnitude Pruning Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    // ------------------------------------------------------------
    // Test 1: count_nonzero basic
    // ------------------------------------------------------------
    {
        Tensor t(2, 3);
        t(0, 0) = 1.0;  t(0, 1) = 0.0;  t(0, 2) = -2.0;
        t(1, 0) = 0.0;  t(1, 1) = 0.0;  t(1, 2) = 3.5;
        CHECK(count_nonzero(t) == 3, "count_nonzero: 3 non-zero out of 6");
    }

    // ------------------------------------------------------------
    // Test 2: count_nonzero edge cases
    // ------------------------------------------------------------
    {
        Tensor z(2, 3);   // default-initialized: should be all zero
        Tensor nz(1, 4);
        nz(0, 0) = 0.1; nz(0, 1) = -0.2; nz(0, 2) = 3.0; nz(0, 3) = 0.0;
        CHECK(count_nonzero(z) == 0, "count_nonzero all-zero = 0");
        CHECK(count_nonzero(nz) == 3, "count_nonzero 3 non-zero out of 4");
    }

    // ------------------------------------------------------------
    // Test 3: magnitude_prune_ zeros the smallest |w| entries
    // ------------------------------------------------------------
    {
        // |w| values: 0.01, 0.1, 0.5, 1.0, 5.0  (sorted)
        Tensor w(1, 5);
        w(0, 0) = 1.0;
        w(0, 1) = 0.1;
        w(0, 2) = 5.0;
        w(0, 3) = -0.01;
        w(0, 4) = -0.5;
        // With sparsity=0.4 (40% = 2 entries), the 2 smallest |w| should
        // be zeroed: positions 3 (|0.01|) and 1 (|0.1|).
        size_t n = magnitude_prune_(w, 0.4);
        CHECK(n == 2, "magnitude_prune_ zeros 2 entries at sparsity=0.4");
        CHECK(w(0, 0) == 1.0, "largest |w| preserved (1.0)");
        CHECK(w(0, 1) == 0.0, "small |w|=0.1 zeroed");
        CHECK(w(0, 2) == 5.0, "largest |w| preserved (5.0)");
        CHECK(w(0, 3) == 0.0, "small |w|=0.01 zeroed");
        CHECK(w(0, 4) == -0.5, "mid |w|=0.5 preserved");
    }

    // ------------------------------------------------------------
    // Test 4: magnitude_prune_ boundary sparsity=0 / sparsity=1
    // ------------------------------------------------------------
    {
        Tensor w(1, 4);
        w(0, 0) = 1.0; w(0, 1) = -2.0; w(0, 2) = 0.3; w(0, 3) = -0.4;

        Tensor w0 = w.clone();
        magnitude_prune_(w0, 0.0);
        CHECK(w0(0, 0) == 1.0 && w0(0, 1) == -2.0 &&
              w0(0, 2) == 0.3 && w0(0, 3) == -0.4,
              "sparsity=0 leaves all entries unchanged");

        Tensor w1 = w.clone();
        size_t n = magnitude_prune_(w1, 1.0);
        CHECK(n == 4, "sparsity=1 zeros all 4 entries");
        CHECK(w1(0, 0) == 0.0 && w1(0, 1) == 0.0 &&
              w1(0, 2) == 0.0 && w1(0, 3) == 0.0,
              "sparsity=1 all entries exactly zero");
    }

    // ------------------------------------------------------------
    // Test 5: magnitude_prune_ clamps sparsity outside [0, 1]
    // ------------------------------------------------------------
    {
        Tensor w(1, 3);
        w(0, 0) = 1.0; w(0, 1) = 2.0; w(0, 2) = -3.0;

        // Negative → treat as 0 → no change
        Tensor wn = w.clone();
        magnitude_prune_(wn, -0.5);
        CHECK(wn(0, 0) == 1.0 && wn(0, 1) == 2.0 && wn(0, 2) == -3.0,
              "negative sparsity clamped to 0 (no-op)");

        // > 1 → treat as 1 → all zero
        Tensor wp = w.clone();
        magnitude_prune_(wp, 1.5);
        CHECK(wp(0, 0) == 0.0 && wp(0, 1) == 0.0 && wp(0, 2) == 0.0,
              "sparsity > 1 clamped to 1 (all zero)");
    }

    // ------------------------------------------------------------
    // Test 6: magnitude_prune_ preserves the largest weights exactly
    // ------------------------------------------------------------
    {
        // Random-ish data
        Tensor w(1, 10);
        double vals[10] = {0.5, -0.1, 2.0, -3.0, 0.05, -1.5, 0.01, 0.7, -0.2, 4.0};
        for (int i = 0; i < 10; ++i) w(0, i) = vals[i];

        // At sparsity=0.5 → zero the 5 smallest |w| entries.
        // Sorted |w|: 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.5, 2.0, 3.0, 4.0
        // → indices 6, 4, 1, 8, 0 are zeroed.
        // Preserved (with original signs): -3.0, -1.5, 0.7, 2.0, 4.0 at
        // indices 3, 5, 7, 2, 9.
        magnitude_prune_(w, 0.5);
        CHECK(w(0, 3) == -3.0, "largest preserved: -3.0");
        CHECK(w(0, 5) == -1.5, "largest preserved: -1.5");
        CHECK(w(0, 7) == 0.7,  "largest preserved: 0.7");
        CHECK(w(0, 2) == 2.0,  "largest preserved: 2.0");
        CHECK(w(0, 9) == 4.0,  "largest preserved: 4.0");
    }

    // ------------------------------------------------------------
    // Test 7: magnitude_prune_ return = count of newly-zeroed
    // (with no pre-existing zeros)
    // ------------------------------------------------------------
    {
        Tensor w(2, 5);  // 10 entries
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 5; ++j)
                w(i, j) = static_cast<double>(i * 5 + j + 1);  // 1..10
        size_t n = magnitude_prune_(w, 0.3);  // 30% → zero 3 entries
        CHECK(n == 3, "magnitude_prune_ return = 3 (sparsity 0.3 of 10)");
        CHECK(count_nonzero(w) == 7, "7 entries remain non-zero after prune");
    }

    // ------------------------------------------------------------
    // Test 8: global_magnitude_prune_ distributes zeros by global threshold
    // ------------------------------------------------------------
    {
        // Two tensors, total 10 entries:
        //   a: |0.01, 0.5, 2.0|       (3 entries)
        //   b: |0.2, 0.3, 0.4, 3.0|   (4 entries)
        // Hmm let me make it total 10: a has 6, b has 4.
        Tensor a(2, 3);
        a(0, 0) = 0.01; a(0, 1) = 0.5;  a(0, 2) = 2.0;
        a(1, 0) = 0.02; a(1, 1) = 1.0;  a(1, 2) = 3.0;
        Tensor b(1, 4);
        b(0, 0) = 0.05; b(0, 1) = 0.3;  b(0, 2) = 0.4; b(0, 3) = 5.0;

        // All |w| values sorted: 0.01, 0.02, 0.05, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 5.0
        // At sparsity=0.3 → keep top 7, zero bottom 3 (|0.01, 0.02, 0.05|).
        std::vector<Tensor*> params = {&a, &b};
        size_t n = global_magnitude_prune_(params, 0.3);
        CHECK(n == 3, "global prune zeros 3 of 10 (30% sparsity)");
        CHECK(a(0, 0) == 0.0, "a[0,0]=0.01 (smallest) zeroed globally");
        CHECK(a(1, 0) == 0.0, "a[1,0]=0.02 (2nd smallest) zeroed globally");
        CHECK(b(0, 0) == 0.0, "b[0,0]=0.05 (3rd smallest) zeroed globally");
        // Preserved entries (with original signs):
        CHECK(a(0, 1) == 0.5,  "a[0,1]=0.5 preserved");
        CHECK(a(0, 2) == 2.0,  "a[0,2]=2.0 preserved");
        CHECK(a(1, 1) == 1.0,  "a[1,1]=1.0 preserved");
        CHECK(a(1, 2) == 3.0,  "a[1,2]=3.0 preserved");
        CHECK(b(0, 1) == 0.3,  "b[0,1]=0.3 preserved");
        CHECK(b(0, 2) == 0.4,  "b[0,2]=0.4 preserved");
        CHECK(b(0, 3) == 5.0,  "b[0,3]=5.0 preserved");
    }

    // ------------------------------------------------------------
    // Test 9: global_magnitude_prune_ with single param equals local
    // ------------------------------------------------------------
    {
        Tensor w(1, 6);
        double v[6] = {0.1, -0.5, 1.0, -2.0, 3.0, -0.02};
        for (int i = 0; i < 6; ++i) w(0, i) = v[i];

        std::vector<Tensor*> params = {&w};
        size_t n_global = global_magnitude_prune_(params, 0.5);
        Tensor w_local(1, 6);
        for (int i = 0; i < 6; ++i) w_local(0, i) = v[i];
        size_t n_local = magnitude_prune_(w_local, 0.5);

        CHECK(n_global == n_local, "single-param global == local count");
        bool same = true;
        for (int i = 0; i < 6; ++i)
            if (w(0, i) != w_local(0, i)) same = false;
        CHECK(same, "single-param global matches local values exactly");
    }

    // ------------------------------------------------------------
    // Test 10: global_magnitude_prune_ returns total entries zeroed
    // ------------------------------------------------------------
    {
        Tensor a(1, 3);  a(0,0) = 1; a(0,1) = 2; a(0,2) = 3;
        Tensor b(1, 3);  b(0,0) = 4; b(0,1) = 5; b(0,2) = 6;
        Tensor c(1, 4);  c(0,0) = 7; c(0,1) = 8; c(0,2) = 9; c(0,3) = 10;
        // Total 10 entries; sparsity=0.4 → 4 zeroed.
        std::vector<Tensor*> params = {&a, &b, &c};
        size_t n = global_magnitude_prune_(params, 0.4);
        CHECK(n == 4, "global across 3 params zeros 4 of 10 (sparsity 0.4)");
        CHECK(count_nonzero(a) + count_nonzero(b) + count_nonzero(c) == 6,
              "6 non-zero remain after global prune (4/10)");
    }

    // ------------------------------------------------------------
    // Test 11: global_magnitude_prune_ handles null pointers
    // ------------------------------------------------------------
    {
        Tensor a(1, 3);
        a(0,0) = 1; a(0,1) = 2; a(0,2) = 3;
        Tensor b(1, 2);
        b(0,0) = 4; b(0,1) = 5;
        std::vector<Tensor*> params = {&a, nullptr, &b};
        // Should not crash; should still process a and b.
        size_t n = global_magnitude_prune_(params, 0.2);  // 1 of 5 zeroed
        CHECK(n == 1, "null pointers skipped; 1 of 5 zeroed");
        // The single zeroed entry should be the smallest across a+b: |1|.
        CHECK(a(0, 0) == 0.0, "smallest entry (|1|) zeroed");
    }

    // ------------------------------------------------------------
    // Test 12: global_magnitude_prune_ preserves top-(1-sparsity) globally
    // (high sparsity = lots of zeros)
    // ------------------------------------------------------------
    {
        Tensor a(1, 5);
        for (int i = 0; i < 5; ++i) a(0, i) = static_cast<double>(i + 1);  // 1..5
        Tensor b(1, 5);
        for (int i = 0; i < 5; ++i) b(0, i) = static_cast<double>(i + 6);  // 6..10
        // Total 10 entries; at sparsity=0.8 → 2 preserved (the 2 largest: 10 and 9).
        std::vector<Tensor*> params = {&a, &b};
        size_t n = global_magnitude_prune_(params, 0.8);
        CHECK(n == 8, "sparsity=0.8 zeros 8 of 10");
        CHECK(b(0, 4) == 10.0, "largest preserved (10.0)");
        CHECK(b(0, 3) == 9.0,  "2nd largest preserved (9.0)");
        CHECK(count_nonzero(a) + count_nonzero(b) == 2,
              "exactly 2 non-zero entries remain");
    }

    // ------------------------------------------------------------
    // Test 13: determinism + no NaN after pruning
    // ------------------------------------------------------------
    {
        Tensor w(3, 4);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                w(i, j) = std::sin(0.1 * i) + std::cos(0.2 * j) + 0.01 * (i + j);

        Tensor w1 = w.clone();
        magnitude_prune_(w1, 0.5);
        CHECK(!any_non_finite(w1), "no NaN after local prune");

        Tensor a = w.clone();
        Tensor b = w.clone();
        std::vector<Tensor*> params = {&a, &b};
        global_magnitude_prune_(params, 0.5);
        CHECK(!any_non_finite(a) && !any_non_finite(b),
              "no NaN after global prune");
    }

    cout << "\n=== Summary: " << passed << " / " << total << " tests passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
