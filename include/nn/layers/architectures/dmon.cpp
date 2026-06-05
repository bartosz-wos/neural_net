#include "dmon.h"
#include <cmath>
#include <algorithm>
#include <vector>

// ============================================================================
// DMonLayer
// ============================================================================
//
// Forward: for each scale tau in [tau_0, ..., tau_{K-1}], compute
//   T_k = exp(tau_k * (A_norm - I))  via truncated Taylor series
//   Z_k = T_k @ X
// then concatenate [Z_0, Z_1, ..., Z_{K-1}] along cols and pass through
// a Dense to produce the output.
//
// Backward:
//   grad_X  = sum_k T_k^T @ grad_Z_k
//   grad_T_k = grad_Z_k @ X^T
//   grad_W_out, grad_b_out from the final dense.
//
// We don't backprop through scales by default (grad_scales_ stays 0) — the
// scales are treated as fixed hyperparameters, which matches the standard
// DMon setup. They can be enabled in a future variant by accumulating
// grad_scales_.
// ============================================================================

DMonLayer::DMonLayer(size_t in_features, size_t out_features,
                     size_t num_scales, size_t taylor_terms,
                     const std::vector<double>& initial_scales)
    : in_features_(in_features), out_features_(out_features),
      num_scales_(num_scales), taylor_terms_(taylor_terms),
      output_dense_(in_features * num_scales, out_features),
      last_concat_(1, 1), last_output_(1, 1), num_nodes_(0)
{
    // Initialize scales
    if (!initial_scales.empty()) {
        scales_ = initial_scales;
    } else {
        // Default geometric schedule: tau_k = 1.5^k for k=0..K-1.
        // The DMon paper recommends a geometric schedule for undirected graphs.
        scales_.resize(num_scales_);
        for (size_t k = 0; k < num_scales_; ++k) {
            scales_[k] = std::pow(1.5, static_cast<double>(k));
        }
    }
    grad_scales_.assign(num_scales_, 0.0);
    output_dense_.init_weights("xavier");
}

void DMonLayer::normalize_adjacency(const Tensor& adj) {
    // Symmetric normalized adjacency with self-loops:
    //   A_norm = D^{-1/2} (A + I) D^{-1/2}
    size_t N = adj.rows;
    Tensor d(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j) deg += adj(i, j);
        deg += 1.0;  // self-loop
        d(i, 0) = (deg > 0.0) ? 1.0 / std::sqrt(deg) : 0.0;
    }
    adj_norm_ = Tensor(N, N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double aij = (i == j) ? adj(i, j) + 1.0 : adj(i, j);
            adj_norm_(i, j) = d(i, 0) * aij * d(j, 0);
        }
    }
}

Tensor DMonLayer::compute_heat_kernel(double tau) const {
    // Heat kernel: H(tau) = exp(tau * (A_norm - I))
    // Truncated Taylor series:
    //   H ≈ sum_{r=0..R} (tau^r / r!) * (A_norm - I)^r
    // where (A_norm - I)^0 = I.
    //
    // We precompute (A_norm - I)^r iteratively: P_{r+1} = P_r @ M, starting
    // with P_0 = I. Then H = sum_{r=0..R} (tau^r / r!) * P_r.
    size_t N = adj_norm_.rows;
    Tensor M(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            M(i, j) = adj_norm_(i, j) - (i == j ? 1.0 : 0.0);

    Tensor H = Tensor::zeros(N, N);
    Tensor P = Tensor::zeros(N, N);  // P_r
    for (size_t i = 0; i < N; ++i) P(i, i) = 1.0;

    double tau_pow = 1.0;    // tau^0
    double factorial = 1.0;  // 0!
    for (size_t r = 0; r < taylor_terms_; ++r) {
        double coef = tau_pow / factorial;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < N; ++j)
                H(i, j) += coef * P(i, j);
        // Prepare next iteration
        tau_pow *= tau;
        factorial *= static_cast<double>(r + 1);
        if (r + 1 < taylor_terms_) {
            // P_{r+1} = P_r @ M
            Tensor P_next(N, N);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j) {
                    double s = 0.0;
                    for (size_t k = 0; k < N; ++k) s += P(i, k) * M(k, j);
                    P_next(i, j) = s;
                }
            P = P_next;
        }
    }
    return H;
}

Tensor DMonLayer::forward(const Tensor& input) {
    (void)input;
    return last_output_;
}

Tensor DMonLayer::forward_with_adj(const Tensor& input, const Tensor& adj) {
    // input: (N, in_features)
    // adj:   (N, N) — adjacency (no self-loops; we add them internally)
    num_nodes_ = input.rows;
    size_t N = num_nodes_;
    size_t F = input.cols;
    last_input_ = input.clone();
    adj_ = adj;

    // 1) Build symmetric normalized adjacency with self-loops.
    normalize_adjacency(adj);

    // 2) For each scale, compute T_k and Z_k = T_k @ X.
    T_k_.clear();
    T_k_.reserve(num_scales_);
    Z_k_.clear();
    Z_k_.reserve(num_scales_);

    // Per-feature concatenation: along the column axis.
    // To do that, we first build a flat (N, K*F) tensor by appending
    // columns of each Z_k to the row.
    last_concat_ = Tensor(N, num_scales_ * F);

    for (size_t k = 0; k < num_scales_; ++k) {
        Tensor T = compute_heat_kernel(scales_[k]);
        T_k_.push_back(T);
        // Z_k = T @ X  → (N, F)
        Tensor Z = T * input;
        Z_k_.push_back(Z);
        // Copy into concat columns [k*F, (k+1)*F)
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < F; ++j)
                last_concat_(i, k * F + j) = Z(i, j);
    }

    // 3) Output dense: (N, K*F) -> (N, out_features)
    Tensor out = output_dense_.forward(last_concat_);
    last_output_ = out;
    return out;
}

Tensor DMonLayer::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;  // not used — handled inside Dense
    // grad_output: (N, out_features)
    // Forward chain:
    //   concat = [Z_0, ..., Z_{K-1}],  Z_k = T_k @ X
    //   out    = W @ concat + b          (Dense forward, internally transpose)
    // Backward:
    //   1) grad_concat from Dense
    //   2) For each scale: grad_Z_k = grad_concat[:, k*F:(k+1)*F]
    //   3) grad_T_k = grad_Z_k @ X^T
    //   4) grad_X  += T_k^T @ grad_Z_k
    //   5) (Optional) grad_tau_k from dH/dtau = sum_r (tau^{r-1} / (r-1)!) * (A_norm - I)^{r-1}
    //      — we skip this; scales are fixed hyperparameters.

    size_t N = last_input_.rows;
    size_t F = in_features_;
    size_t K = num_scales_;

    // 1) Backprop through output dense.
    // grad_concat = (N, K*F)
    Tensor grad_concat = output_dense_.backward(grad_output, 0.0);

    // 2-4) Accumulate grad_input.
    Tensor grad_input(N, F);
    grad_input.fill(0.0);

    for (size_t k = 0; k < K; ++k) {
        // Extract grad_Z_k = grad_concat[:, k*F:(k+1)*F)
        Tensor grad_Z(N, F);
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < F; ++j)
                grad_Z(i, j) = grad_concat(i, k * F + j);

        // grad_input += T_k^T @ grad_Z   (T is symmetric, so T^T = T)
        Tensor Tt = T_k_[k].transpose();
        Tensor contrib = Tt * grad_Z;
        grad_input += contrib;
    }

    return grad_input;
}

void DMonLayer::update_weights(double learning_rate) {
    output_dense_.update_weights(learning_rate);
}

void DMonLayer::zero_grad() {
    output_dense_.zero_grad();
}

std::vector<Tensor*> DMonLayer::parameters() {
    return output_dense_.parameters();
}

std::vector<Tensor*> DMonLayer::gradients() {
    return output_dense_.gradients();
}

// ============================================================================
// DMonModel
// ============================================================================

DMonModel::DMonModel(size_t num_nodes, size_t in_features, size_t hidden_dim,
                     size_t out_features, size_t num_layers,
                     size_t num_scales, size_t taylor_terms,
                     const std::vector<double>& initial_scales)
    : num_nodes_(num_nodes), in_features_(in_features),
      hidden_dim_(hidden_dim), out_features_(out_features),
      num_layers_(num_layers), num_scales_(num_scales),
      taylor_terms_(taylor_terms), initial_scales_(initial_scales),
      input_proj_(in_features, hidden_dim),
      classifier_(hidden_dim, out_features),
      last_input_(1, in_features), last_input_proj_(1, hidden_dim),
      adj_(num_nodes, num_nodes)
{
    input_proj_.init_weights("xavier");
    classifier_.init_weights("xavier");
    dmon_layers_.clear();
    dmon_layers_.reserve(num_layers_);
    for (size_t l = 0; l < num_layers_; ++l) {
        dmon_layers_.emplace_back(hidden_dim, hidden_dim,
                                  num_scales_, taylor_terms_, initial_scales_);
    }
    layer_outputs_pre_relu_.clear();
    layer_outputs_pre_relu_.reserve(num_layers_);
}

Tensor DMonModel::forward(const Tensor& input) {
    (void)input;
    // Real forward is in forward_with_adj; this stub returns the last
    // cached output for callers that don't have an adjacency matrix.
    static Tensor dummy(1, 1);
    return dummy;
}

Tensor DMonModel::forward_with_adj(const Tensor& input, const Tensor& adj) {
    // input: (N, in_features)
    // adj:   (N, N) — adjacency (no self-loops)
    last_input_ = input.clone();
    adj_ = adj;

    size_t N = input.rows;
    size_t F = input.cols;
    (void)N; (void)F;  // not used directly; only via layer input

    // Optional input projection (if dims differ)
    Tensor h;
    if (in_features_ == hidden_dim_) {
        h = input.clone();
    } else {
        h = input_proj_.forward(input);
    }
    last_input_proj_ = h.clone();

    layer_outputs_pre_relu_.clear();
    layer_outputs_pre_relu_.reserve(num_layers_);

    for (size_t l = 0; l < num_layers_; ++l) {
        Tensor out = dmon_layers_[l].forward_with_adj(h, adj);
        // Cache the pre-ReLU output for the backward pass (used as ReLU mask)
        layer_outputs_pre_relu_.push_back(out.clone());
        // ReLU between layers
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                if (out(i, j) < 0.0) out(i, j) = 0.0;
        h = out;
    }

    Tensor logits = classifier_.forward(h);
    return logits;
}

Tensor DMonModel::backward(const Tensor& grad_output, double learning_rate) {
    (void)learning_rate;  // not used — handled inside Dense
    // grad_output: (N, out_features)
    // Backward chain:
    //   logits -> classifier -> ReLU -> dmon_layers[last] -> ... -> dmon_layers[0]
    //   -> input_proj -> input
    //
    // Each DMonLayer returns the grad of the layer INPUT (pre-ReLU) so
    // we apply ReLU mask here to grad_X before passing to the next layer.
    // 1) Backprop through classifier.
    Tensor grad_h = classifier_.backward(grad_output, 0.0);

    // 2) Backprop through DMon layers (last to first), with ReLU mask
    //    between layers.
    for (int l = static_cast<int>(num_layers_) - 1; l >= 0; --l) {
        // grad_h is dL/d(post_relu_l) = dL/d(DMonLayer_l_output) — but
        // we need to apply the ReLU mask first. The pre-ReLU output of
        // layer l is cached in layer_outputs_pre_relu_[l].
        const Tensor& pre = layer_outputs_pre_relu_[l];
        for (size_t i = 0; i < grad_h.rows; ++i)
            for (size_t j = 0; j < grad_h.cols; ++j)
                if (pre(i, j) <= 0.0) grad_h(i, j) = 0.0;

        // Backward through DMon layer → grad of its input.
        // The DMon layer applies its own final dense backward.
        grad_h = dmon_layers_[l].backward(grad_h, 0.0);
    }

    // 3) Backprop through input projection.
    if (in_features_ != hidden_dim_) {
        grad_h = input_proj_.backward(grad_h, 0.0);
    }
    return grad_h;
}

void DMonModel::update_weights(double learning_rate) {
    input_proj_.update_weights(learning_rate);
    for (auto& l : dmon_layers_) l.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void DMonModel::zero_grad() {
    input_proj_.zero_grad();
    for (auto& l : dmon_layers_) l.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> DMonModel::parameters() {
    std::vector<Tensor*> p;
    auto ip = input_proj_.parameters();
    p.insert(p.end(), ip.begin(), ip.end());
    for (auto& l : dmon_layers_) {
        auto lp = l.parameters();
        p.insert(p.end(), lp.begin(), lp.end());
    }
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> DMonModel::gradients() {
    std::vector<Tensor*> g;
    auto ig = input_proj_.gradients();
    g.insert(g.end(), ig.begin(), ig.end());
    for (auto& l : dmon_layers_) {
        auto lg = l.gradients();
        g.insert(g.end(), lg.begin(), lg.end());
    }
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}
