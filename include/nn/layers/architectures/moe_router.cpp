// moe_router.cpp — pure-stateless top-k softmax routing + load-balance
// utility for Mixture-of-Experts layers. See moe_router.h for spec.

#include "moe_router.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

// partial_sort_desc to find top-k by value
void topk_desc(const std::vector<std::pair<double, size_t>>& scored, size_t k,
               std::vector<size_t>& out_idx) {
    out_idx.resize(k);
    // Build an index vector and use nth_element for O(B * E) instead of O(B E log E).
    std::vector<size_t> idx(scored.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](size_t a, size_t b) {
                          return scored[a].first > scored[b].first;
                      });
    for (size_t i = 0; i < k; ++i) out_idx[i] = idx[i];
}

}  // namespace

MoeRouterState MoeRouter::softmax_topk(const Tensor& logits, size_t k) {
    if (logits.rows == 0) {
        throw std::invalid_argument("MoeRouter::softmax_topk: empty logits (rows=0)");
    }
    size_t B = logits.rows;
    size_t E = static_cast<size_t>(logits.cols);
    if (E == 0) {
        throw std::invalid_argument("MoeRouter::softmax_topk: E (cols) must be > 0");
    }
    if (k == 0) {
        throw std::invalid_argument("MoeRouter::softmax_topk: k must be >= 1");
    }
    if (k > E) {
        throw std::invalid_argument("MoeRouter::softmax_topk: k must be <= E");
    }

    MoeRouterState s;
    s.probs       = Tensor::zeros(B, E);
    s.topk_idx.assign(B, std::vector<size_t>(k, 0));
    s.dispatch_frac.assign(E, 0.0);
    s.mean_gate_prob.assign(E, 0.0);
    s.maxv.assign(B, 0.0);
    s.exps.assign(B, std::vector<double>(k, 0.0));
    s.logsumexp.assign(B, 0.0);
    s.num_experts = E;
    s.k = k;

    // Per-row softmax over the top-k (numerically-stable: max-subtraction).
    for (size_t b = 0; b < B; ++b) {
        // 1) Score every expert and find the top-k via partial_sort.
        std::vector<std::pair<double, size_t>> scored(E);
        for (size_t e = 0; e < E; ++e) {
            scored[e] = {logits(b, e), e};
        }
        std::vector<size_t> sel;
        topk_desc(scored, k, sel);
        for (size_t i = 0; i < k; ++i) s.topk_idx[b][i] = sel[i];

        // 2) maxv for stability
        double maxv = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < k; ++i) {
            const auto& p = scored[sel[i]];
            if (p.first > maxv) maxv = p.first;
        }
        s.maxv[b] = maxv;

        // 3) exp + sum
        double sumexp = 0.0;
        for (size_t i = 0; i < k; ++i) {
            double v = std::exp(scored[sel[i]].first - maxv);
            s.exps[b][i] = v;
            sumexp += v;
        }
        if (sumexp < 1e-30) sumexp = 1e-30;
        s.logsumexp[b] = std::log(sumexp);

        // 4) write probs
        for (size_t i = 0; i < k; ++i) {
            size_t e = sel[i];
            double p = s.exps[b][i] / sumexp;
            s.probs(b, e) = p;
        }
    }

    // dispatch_frac: (# selections per expert) / B
    for (size_t b = 0; b < B; ++b) {
        for (size_t i = 0; i < k; ++i) {
            size_t e = s.topk_idx[b][i];
            s.dispatch_frac[e] += 1.0;
        }
    }
    if (B > 0) {
        for (size_t e = 0; e < E; ++e) {
            s.dispatch_frac[e] /= static_cast<double>(B);
        }
    }

    // mean_gate_prob: (1/B) * Σ_b probs[b, e]
    for (size_t b = 0; b < B; ++b) {
        for (size_t e = 0; e < E; ++e) {
            s.mean_gate_prob[e] += s.probs(b, e);
        }
    }
    if (B > 0) {
        for (size_t e = 0; e < E; ++e) {
            s.mean_gate_prob[e] /= static_cast<double>(B);
        }
    }

    return s;
}

Tensor MoeRouter::softmax_topk_backward(const Tensor& d_gate_probs,
                                          const MoeRouterState& state) {
    size_t B = state.probs.rows;
    size_t E = state.probs.cols;
    if ((size_t)d_gate_probs.rows != B || (size_t)d_gate_probs.cols != E) {
        throw std::invalid_argument(
            "MoeRouter::softmax_topk_backward: d_gate_probs shape mismatch");
    }
    size_t k = state.k;

    Tensor d_logits = Tensor::zeros(B, E);

    for (size_t b = 0; b < B; ++b) {
        // s = Σ_{e in S} probs[b, e] * d_gate_probs[b, e]
        double s = 0.0;
        for (size_t i = 0; i < k; ++i) {
            size_t e = state.topk_idx[b][i];
            s += state.probs(b, e) * d_gate_probs(b, e);
        }
        for (size_t i = 0; i < k; ++i) {
            size_t e = state.topk_idx[b][i];
            d_logits(b, e) = state.probs(b, e) * (d_gate_probs(b, e) - s);
        }
        // Non-selected entries stay zero (hard topk mask).
    }

    return d_logits;
}

double MoeRouter::load_balance_loss_unscaled(
    const std::vector<double>& dispatch_frac,
    const std::vector<double>& mean_gate_prob) {
    size_t E = dispatch_frac.size();
    if (E == 0) {
        throw std::invalid_argument("MoeRouter::load_balance_loss_unscaled: E must be > 0");
    }
    if (mean_gate_prob.size() != E) {
        throw std::invalid_argument(
            "MoeRouter::load_balance_loss_unscaled: dispatch_frac and mean_gate_prob sizes differ");
    }
    double sum = 0.0;
    for (size_t e = 0; e < E; ++e) {
        sum += dispatch_frac[e] * mean_gate_prob[e];
    }
    return static_cast<double>(E) * sum;
}
