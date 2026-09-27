#include "tome.h"
#include <cmath>
#include <stdexcept>
#include <limits>

TokenMerging::TokenMerging(double reduce_ratio)
    : reduce_ratio_(reduce_ratio), last_N_(0), last_d_(0), last_N_out_(0) {
    if (reduce_ratio_ <= 0.0 || reduce_ratio_ > 1.0) {
        throw std::invalid_argument("TokenMerging: ratio must be in (0, 1]");
    }
}

size_t TokenMerging::n_out(size_t N) const {
    size_t n = (size_t)std::floor((double)N * reduce_ratio_);
    if (n == 0 && N > 0) n = 1;  // never collapse to zero unless N=0
    return n;
}

static double safe_norm(const Tensor& x, size_t row, size_t d) {
    double s = 0.0;
    for (size_t j = 0; j < d; ++j) {
        double v = x(row, j);
        s += v * v;
    }
    return std::sqrt(s);
}

static void compute_match_edges(const Tensor& x, size_t /*N*/, size_t d,
                                 size_t nA, size_t nB,
                                 std::vector<std::pair<size_t, size_t>>& edges_out) {
    // For each a in A (even indices), pick argmax_b s_ab.
    // s_ab = (x_a · x_b) / (2 * ||x_a|| * ||x_b|| + eps)
    edges_out.clear();
    edges_out.reserve(nA);
    constexpr double eps = 1e-8;
    for (size_t ai = 0; ai < nA; ++ai) {
        size_t a = 2 * ai;  // a-index = 2 * ai
        double na = safe_norm(x, a, d);
        if (na < eps) {
            // Zero-norm row: tie-break by smallest b (paper convention
            // picks by index order). Choose b=0.
            edges_out.emplace_back(a, 0);
            continue;
        }
        double best_score = -std::numeric_limits<double>::infinity();
        size_t best_b = 0;
        for (size_t bj = 0; bj < nB; ++bj) {
            size_t b = 2 * bj + 1;  // b-index = 2 * bj + 1
            double nb = safe_norm(x, b, d);
            double score;
            if (nb < eps) {
                score = -std::numeric_limits<double>::infinity();
            } else {
                double dot = 0.0;
                for (size_t j = 0; j < d; ++j) dot += x(a, j) * x(b, j);
                score = dot / (2.0 * na * nb + eps);
            }
            // tie-break: prefer SMALLER bj (smaller index)
            if (score > best_score) {
                best_score = score;
                best_b = b;
            }
        }
        edges_out.emplace_back(a, best_b);
    }
}

Tensor TokenMerging::forward(const Tensor& input) {
    size_t N = input.rows;
    size_t d = input.cols;
    size_t n_out_ = n_out(N);

    // Partition
    size_t nA = (N + 1) / 2;  // even indices 0,2,...
    size_t nB = N / 2;        // odd  indices 1,3,...

    // Compute match edges
    compute_match_edges(input, N, d, nA, nB, last_edges_);

    // n_keep = min(n_out_, nA). For ratio in (0, 1], n_out_ <= nA always holds
    // (nA = ceil(N/2), n_out_ = floor(N*r) <= N/2 <= nA), but be safe.
    size_t n_keep = std::min(n_out_, last_edges_.size());

    Tensor y(n_keep, d);
    for (size_t k = 0; k < n_keep; ++k) {
        size_t a = last_edges_[k].first;
        size_t b = last_edges_[k].second;
        for (size_t j = 0; j < d; ++j) {
            y(k, j) = 0.5 * (input(a, j) + input(b, j));
        }
    }

    last_N_ = N;
    last_d_ = d;
    last_N_out_ = n_keep;
    return y;
}

Tensor TokenMerging::forward_with_unmerge(const Tensor& input) {
    return forward(input);
}

Tensor TokenMerging::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t d = last_d_;
    Tensor grad_input = Tensor::zeros(last_N_, d);
    for (size_t k = 0; k < last_N_out_; ++k) {
        size_t a = last_edges_[k].first;
        size_t b = last_edges_[k].second;
        for (size_t j = 0; j < d; ++j) {
            grad_input(a, j) += 0.5 * grad_output(k, j);
            grad_input(b, j) += 0.5 * grad_output(k, j);
        }
    }
    return grad_input;
}

Tensor TokenMerging::unmerge(const Tensor& grad_y) {
    size_t d = last_d_;
    Tensor grad_x = Tensor::zeros(last_N_, d);
    for (size_t k = 0; k < last_N_out_; ++k) {
        size_t a = last_edges_[k].first;
        size_t b = last_edges_[k].second;
        for (size_t j = 0; j < d; ++j) {
            grad_x(a, j) += 0.5 * grad_y(k, j);
            grad_x(b, j) += 0.5 * grad_y(k, j);
        }
    }
    return grad_x;
}

void TokenMerging::update_weights(double /* learning_rate */) {}
void TokenMerging::zero_grad() {}

std::vector<Tensor*> TokenMerging::parameters() { return {}; }
std::vector<Tensor*> TokenMerging::gradients() { return {}; }
