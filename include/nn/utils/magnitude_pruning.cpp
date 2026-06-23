#include "magnitude_pruning.h"
#include <algorithm>
#include <cmath>

// Count entries strictly non-zero.
size_t count_nonzero(const Tensor& t) {
    size_t n = 0;
    for (size_t i = 0; i < t.data.size(); ++i)
        if (t.data[i] != 0.0) ++n;
    return n;
}

// Magnitude-prune a single weight tensor (in-place).
//
// Strategy: collect all |w| values, sort them, and zero out the
// `floor(sparsity * N)` smallest. We compute the threshold as the
// largest-magnitude entry among the entries-to-zero. Equivalently:
// the threshold is the (k-th smallest of the sorted magnitudes) where
// k = floor(sparsity * N). Anything strictly below that threshold is
// zeroed. (Entries equal to the threshold but in the "keep" half are
// preserved — we only zero the smallest `k`, not all entries below
// the k-th value, to avoid ambiguity at the threshold.)
//
// Returns: number of entries actually zeroed (could be less than
// requested if some entries were already zero).
//
// Edge cases:
//   * sparsity clamped to [0, 1].
//   * Empty tensor returns 0.
//   * sparsity=0 → no change, return 0.
//   * sparsity=1 → zero every non-zero entry (and zero of any all-
//     zero entries), return the count of zeroed entries.
size_t magnitude_prune_(Tensor& w, double sparsity) {
    // Clamp
    if (sparsity < 0.0) sparsity = 0.0;
    if (sparsity > 1.0) sparsity = 1.0;

    const size_t N = w.data.size();
    if (N == 0 || sparsity == 0.0) return 0;

    // Collect magnitudes.
    std::vector<double> mags;
    mags.reserve(N);
    for (size_t i = 0; i < N; ++i) mags.push_back(std::fabs(w.data[i]));

    // Number of entries to zero (round down — fractional counts can't
    // be sensibly distributed).
    size_t k = static_cast<size_t>(std::floor(sparsity * static_cast<double>(N)));
    if (k == 0) return 0;

    // Partial-sort: find the k-th smallest magnitude. After nth_element,
    // mags[k-1] is the threshold. Entries with |w| < threshold are zeroed;
    // entries with |w| == threshold that appear among the bottom-k are
    // zeroed as well (and there are exactly k of them, by construction).
    //
    // Note: nth_element leaves the first k elements in arbitrary order
    // but all <= elements from k onwards. We want exactly the k smallest,
    // so after nth_element we zero entries with |w| <= mags[k-1] (the
    // threshold), but we need to be careful: there might be more than k
    // entries tied at the threshold. We just zero the FIRST k of the
    // [0, k) range. Easier approach: sort fully if N is small, else
    // nth_element and zero entries with index in [0, k).
    //
    // Simplest correct implementation: full sort. For ML params this
    // is rarely the bottleneck — the user typically calls prune once
    // per training step, not in the hot path.
    std::vector<double> sorted = mags;  // copy for sorting (keep mags for zero pass)
    std::sort(sorted.begin(), sorted.end());

    // Threshold = the k-th smallest magnitude.
    double thresh = sorted[k - 1];

    // Zero the first k entries (those with |w| <= thresh after sorting).
    // After sorting, the first k entries are exactly the k smallest.
    // To find the corresponding original entries, we can iterate over
    // the original mags and zero those with value <= thresh, but we
    // need to handle ties at the threshold correctly (only zero k of them).
    //
    // Approach: zero any entry with |w| < thresh, plus enough
    // entries with |w| == thresh to reach k total.
    size_t zeroed = 0;
    size_t needed_at_thresh = 0;
    size_t already_below = 0;
    size_t at_thresh = 0;
    for (size_t i = 0; i < N; ++i) {
        if (mags[i] < thresh) ++already_below;
        else if (mags[i] == thresh) ++at_thresh;
    }
    needed_at_thresh = (already_below < k) ? (k - already_below) : 0;
    // Cap at total at_thresh (could be more than needed due to ties)
    if (needed_at_thresh > at_thresh) needed_at_thresh = at_thresh;

    for (size_t i = 0; i < N; ++i) {
        if (mags[i] < thresh) {
            w.data[i] = 0.0;
            ++zeroed;
        } else if (mags[i] == thresh && needed_at_thresh > 0) {
            w.data[i] = 0.0;
            ++zeroed;
            --needed_at_thresh;
        }
    }
    return zeroed;
}

// Magnitude-prune globally across a list of parameter tensors.
//
// Strategy: collect all (param_index, entry_index, |value|) triples
// from all params. Sort by |value| ascending. Take the smallest
// floor(sparsity * total_N). Walk back through the params and zero
// the corresponding entries in-place.
//
// Returns: total entries zeroed.
size_t global_magnitude_prune_(const std::vector<Tensor*>& params, double sparsity) {
    // Clamp
    if (sparsity < 0.0) sparsity = 0.0;
    if (sparsity > 1.0) sparsity = 1.0;

    // First pass: collect all (flat_index_in_param, |value|) triples,
    // and total entry count (skipping nulls).
    struct Entry {
        size_t param_idx;     // index into params
        size_t flat_idx;      // index into data vector
        double mag;           // |value|
    };
    std::vector<Entry> entries;
    size_t total_N = 0;
    for (size_t pi = 0; pi < params.size(); ++pi) {
        Tensor* p = params[pi];
        if (!p) continue;
        const size_t M = p->data.size();
        total_N += M;
        for (size_t i = 0; i < M; ++i) {
            entries.push_back({pi, i, std::fabs(p->data[i])});
        }
    }
    if (total_N == 0 || sparsity == 0.0) return 0;

    size_t k = static_cast<size_t>(std::floor(sparsity * static_cast<double>(total_N)));
    if (k == 0) return 0;

    // Sort ascending by |value|.
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.mag < b.mag; });

    // Threshold = the k-th smallest magnitude.
    double thresh = entries[k - 1].mag;

    // Zero entries: any entry with mag < thresh, plus enough at-thresh
    // entries to reach k total.
    size_t already_below = 0;
    size_t at_thresh = 0;
    for (size_t i = 0; i < k; ++i) {
        if (entries[i].mag < thresh) ++already_below;
        else ++at_thresh;  // == thresh (we sorted the first k)
    }
    // Note: entries[k..] have mag >= thresh, so all `mag < thresh`
    // entries are in entries[0..k).
    size_t needed_at_thresh = (already_below < k) ? (k - already_below) : 0;
    if (needed_at_thresh > at_thresh) needed_at_thresh = at_thresh;

    size_t zeroed = 0;
    for (size_t i = 0; i < k; ++i) {
        if (entries[i].mag < thresh) {
            params[entries[i].param_idx]->data[entries[i].flat_idx] = 0.0;
            ++zeroed;
        } else if (needed_at_thresh > 0) {
            params[entries[i].param_idx]->data[entries[i].flat_idx] = 0.0;
            ++zeroed;
            --needed_at_thresh;
        }
    }
    return zeroed;
}
