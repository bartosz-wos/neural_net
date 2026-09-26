#include "sagpool.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace {
// Helper: softmax over a length-N vector (stored as (N, 1) tensor)
Tensor softmax_over_n(const Tensor& v) {
    size_t N = v.rows;
    Tensor out(N, 1);
    // Subtract max for numerical stability
    double mx = -1e30;
    for (size_t i = 0; i < N; ++i) if (v(i, 0) > mx) mx = v(i, 0);
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        out(i, 0) = std::exp(v(i, 0) - mx);
        sum += out(i, 0);
    }
    if (sum > 0.0) {
        for (size_t i = 0; i < N; ++i) out(i, 0) /= sum;
    } else {
        // Degenerate: all exp values underflowed. Use uniform.
        for (size_t i = 0; i < N; ++i) out(i, 0) = 1.0 / N;
    }
    return out;
}
} // namespace

// ============================================================================
// SAGPool
// ============================================================================

SAGPool::SAGPool(size_t in_features, double ratio, bool use_gcn)
    : in_features_(in_features), ratio_(ratio), use_gcn_(use_gcn) {
    if (in_features == 0)
        throw std::invalid_argument("SAGPool: in_features must be > 0");
    if (ratio <= 0.0 || ratio > 1.0)
        throw std::invalid_argument("SAGPool: ratio must be in (0, 1]");

    // Initialize W_pool with small random values (Xavier-like)
    W_pool_ = Tensor(in_features, 1);
    grad_W_pool_ = Tensor(in_features, 1);
    grad_W_pool_.fill(0.0);
    double scale = std::sqrt(1.0 / static_cast<double>(in_features));
    for (size_t i = 0; i < in_features; ++i)
        W_pool_(i, 0) = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale;

    last_grad_A_ = Tensor(0, 0);
}

size_t SAGPool::num_nodes_out(size_t N) const {
    return static_cast<size_t>(std::ceil(ratio_ * static_cast<double>(N)));
}

Tensor SAGPool::forward(const Tensor& /*input*/) {
    throw std::runtime_error("SAGPool: forward(input) without adj not supported; use forward_with_adj(input, adj)");
}

std::pair<Tensor, Tensor> SAGPool::forward_with_adj(const Tensor& input, const Tensor& adj) {
    size_t N = input.rows;
    size_t in_f = input.cols;
    if (in_f != in_features_)
        throw std::invalid_argument("SAGPool: input feature dim != in_features");
    if (adj.rows != N || adj.cols != N)
        throw std::invalid_argument("SAGPool: adj must be N x N");

    last_input_ = input.clone();
    last_A_ = adj.clone();
    last_N_ = N;
    last_N_out_ = num_nodes_out(N);

    // --- Step 1: Compute A_hat = D^{-1/2}(A + I)D^{-1/2} ---
    Tensor A_plus_I(N, N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A_plus_I(i, j) = adj(i, j) + (i == j ? 1.0 : 0.0);
        }
    }
    // Degree vector (with self-loops included)
    Tensor d_inv_sqrt(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j) deg += A_plus_I(i, j);
        d_inv_sqrt(i, 0) = (deg > 0.0) ? 1.0 / std::sqrt(deg) : 0.0;
    }
    last_A_hat_ = Tensor(N, N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            last_A_hat_(i, j) = d_inv_sqrt(i, 0) * A_plus_I(i, j) * d_inv_sqrt(j, 0);
        }
    }

    // --- Step 2: Z_raw = ReLU(A_hat X W_pool) if use_gcn, else X W_pool ---
    Tensor Z_raw(N, 1);
    if (use_gcn_) {
        // Z_raw[k] = max(0, sum_{k',j} A_hat[k, k'] * X[k', j] * W_pool[j])
        for (size_t k = 0; k < N; ++k) {
            double s = 0.0;
            for (size_t kp = 0; kp < N; ++kp) {
                double row_sum = 0.0;
                for (size_t j = 0; j < in_f; ++j) {
                    row_sum += input(kp, j) * W_pool_(j, 0);
                }
                s += last_A_hat_(k, kp) * row_sum;
            }
            Z_raw(k, 0) = std::max(0.0, s);
        }
    } else {
        // Z_raw[k] = X[k, :] @ W_pool  (no adjacency, no ReLU)
        for (size_t k = 0; k < N; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < in_f; ++j) s += input(k, j) * W_pool_(j, 0);
            Z_raw(k, 0) = s;
        }
    }
    last_Z_raw_ = Z_raw;

    // --- Step 3: Z = softmax(Z_raw) ---
    Tensor Z = softmax_over_n(Z_raw);
    last_Z_ = Z;

    // --- Step 4: top-k selection by Z ---
    // We need to find the N_out indices with the largest Z values.
    // Use pair<Z[k], k> and partial_sort descending.
    std::vector<std::pair<double, size_t>> pairs(N);
    for (size_t k = 0; k < N; ++k) {
        pairs[k] = {Z(k, 0), k};
    }
    // partial_sort puts the top N_out at the front in DESCENDING order of Z.
    // For tie-breaking (equal Z): std::partial_sort is NOT stable, but we
    // can break ties deterministically by sorting the pairs with a comparator
    // that uses (Z desc, index asc). Use full sort — N is small (graphs).
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<double, size_t>& a,
                 const std::pair<double, size_t>& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
              });
    last_idx_.resize(last_N_out_);
    for (size_t i = 0; i < last_N_out_; ++i) last_idx_[i] = pairs[i].second;

    // --- Step 5: X' = (X ⊙ Z)[idx, :] ---
    Tensor Xp(last_N_out_, in_f);
    for (size_t i = 0; i < last_N_out_; ++i) {
        size_t src = last_idx_[i];
        double zk = Z(src, 0);
        for (size_t j = 0; j < in_f; ++j) {
            Xp(i, j) = input(src, j) * zk;
        }
    }

    // --- Step 6: A' = A[idx, idx] ---
    Tensor Ap(last_N_out_, last_N_out_);
    for (size_t i = 0; i < last_N_out_; ++i) {
        for (size_t j = 0; j < last_N_out_; ++j) {
            Ap(i, j) = adj(last_idx_[i], last_idx_[j]);
        }
    }

    return {Xp, Ap};
}

Tensor SAGPool::backward(const Tensor& grad_output, double learning_rate) {
    // backward() treats grad_A as zero. Construct a zero N_out × N_out tensor.
    if (grad_output.rows != last_N_out_ || grad_output.cols != in_features_)
        throw std::invalid_argument("SAGPool backward: grad_output shape mismatch");
    Tensor zero_grad_A(last_N_out_, last_N_out_);
    zero_grad_A.fill(0.0);
    return backward_with_adj_grads(grad_output, zero_grad_A, learning_rate);
}

Tensor SAGPool::backward_with_adj_grads(const Tensor& grad_X, const Tensor& grad_A,
                                         double learning_rate) {
    size_t N = last_N_;
    size_t in_f = in_features_;
    size_t N_out = last_N_out_;

    if (grad_X.rows != N_out || grad_X.cols != in_f)
        throw std::invalid_argument("SAGPool backward: grad_X shape mismatch");
    if (grad_A.rows != N_out || grad_A.cols != N_out)
        throw std::invalid_argument("SAGPool backward: grad_A shape mismatch");

    // Accumulate grad_input (N × in_f), initialized to zero.
    Tensor grad_input(N, in_f);
    grad_input.fill(0.0);

    // --- Step 1+2: Chain from X' = (X ⊙ Z)[idx, :] back to (X, Z). ---
    // For each pooled row i (with source node src = idx[i]):
    //   grad_X_src[i, j] = grad_X'[i, j] * Z[src]
    //   grad_Z[src]      = sum_j grad_X'[i, j] * X[src, j]
    // grad_X_pooled[k, j] accumulates over all i where idx[i] = k (should be at most 1).
    Tensor grad_Z(N, 1);
    grad_Z.fill(0.0);
    for (size_t i = 0; i < N_out; ++i) {
        size_t src = last_idx_[i];
        double zk = last_Z_(src, 0);
        for (size_t j = 0; j < in_f; ++j) {
            double xij = last_input_(src, j);
            grad_input(src, j) += grad_X(i, j) * zk;
            grad_Z(src, 0) += grad_X(i, j) * xij;
        }
    }

    // --- Step 4: softmax backward. ---
    // dL/d(Z_raw[k]) = Z[k] * (dZ[k] - sum_m Z[m] * dZ[m])
    double z_dot_dz = 0.0;
    for (size_t k = 0; k < N; ++k) {
        z_dot_dz += last_Z_(k, 0) * grad_Z(k, 0);
    }
    Tensor grad_Z_raw(N, 1);
    for (size_t k = 0; k < N; ++k) {
        grad_Z_raw(k, 0) = last_Z_(k, 0) * (grad_Z(k, 0) - z_dot_dz);
    }

    // --- Step 5: ReLU backward (only if use_gcn=true). ---
    // For use_gcn=false, we have no ReLU and no A_hat chain.
    Tensor grad_G_pre(N, 1);
    grad_G_pre.fill(0.0);

    if (use_gcn_) {
        for (size_t k = 0; k < N; ++k) {
            grad_G_pre(k, 0) = (last_Z_raw_(k, 0) > 0.0) ? grad_Z_raw(k, 0) : 0.0;
        }
    } else {
        // No ReLU, but also no A_hat — Z_raw = X W_pool directly.
        // Chain through that: dL/d(X W_pool) = grad_Z_raw.
        // dL/d(W_pool[j]) = sum_k X[k, j] * grad_Z_raw[k]
        // dL/d(X[k, j])   = grad_Z_raw[k] * W_pool[j]
        for (size_t j = 0; j < in_f; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) sum += last_input_(k, j) * grad_Z_raw(k, 0);
            grad_W_pool_(j, 0) += sum;
        }
        for (size_t k = 0; k < N; ++k) {
            for (size_t j = 0; j < in_f; ++j) {
                grad_input(k, j) += grad_Z_raw(k, 0) * W_pool_(j, 0);
            }
        }
        // Store grad_A (zero — there's no path from A to anything in use_gcn=false)
        last_grad_A_ = Tensor(N, N);
        last_grad_A_.fill(0.0);
        (void)learning_rate;  // unused (handled by update_weights)
        (void)grad_A;          // unused
        return grad_input;
    }

    // --- Step 6: GCN backward ---
    // Forward: G_pre = A_hat X W_pool (vector, since W_pool is in_f x 1)
    //   G_pre[k] = sum_{k'} A_hat[k, k'] * (sum_j X[k', j] * W_pool[j])
    //            = sum_{k'} A_hat[k, k'] * g[k']
    //   where g[k'] = X @ W_pool ∈ R^N.
    //
    //   dL/d(W_pool[j]) = sum_k dL/d(G_pre[k]) * A_hat[k, k'] * X[k', j]
    //                   = sum_{k'} X[k', j] * (sum_k A_hat[k, k'] * dL/d(G_pre[k]))
    //                   = sum_{k'} X[k', j] * (A_hat^T dL/d(G_pre))[k']
    //                   = X^T @ (A_hat^T @ dL/d(G_pre))
    //   dL/d(X[k', j])   = W_pool[j] * sum_k A_hat[k, k'] * dL/d(G_pre[k])
    //                   = W_pool[j] * (A_hat^T dL/d(G_pre))[k']
    //   dL/d(A_hat[k, k']) = dL/d(G_pre[k]) * g[k']
    //                       = dL/d(G_pre[k]) * G_pre[k']

    // Compute grad_A_hat (N × N)
    Tensor grad_A_hat(N, N);
    grad_A_hat.fill(0.0);
    for (size_t k = 0; k < N; ++k) {
        for (size_t kp = 0; kp < N; ++kp) {
            grad_A_hat(k, kp) = grad_G_pre(k, 0) * last_Z_raw_(k, 0);  // grad_A_hat(k,k') = d_G_pre[k] * G_pre[k']
        }
    }

    // Compute d_X_g (A_hat^T grad_G_pre) ∈ R^N
    Tensor d_X_g(N, 1);
    d_X_g.fill(0.0);
    for (size_t kp = 0; kp < N; ++kp) {
        double s = 0.0;
        for (size_t k = 0; k < N; ++k) s += last_A_hat_(k, kp) * grad_G_pre(k, 0);
        d_X_g(kp, 0) = s;
    }

    // d_W_pool = X^T @ d_X_g
    for (size_t j = 0; j < in_f; ++j) {
        double sum = 0.0;
        for (size_t kp = 0; kp < N; ++kp) sum += last_input_(kp, j) * d_X_g(kp, 0);
        grad_W_pool_(j, 0) += sum;
    }

    // d_X[k, j] += W_pool[j] * d_X_g[k]
    for (size_t k = 0; k < N; ++k) {
        for (size_t j = 0; j < in_f; ++j) {
            grad_input(k, j) += d_X_g(k, 0) * W_pool_(j, 0);
        }
    }

    // --- Step 7: grad_A from grad_A_hat, treating d_inv_sqrt as fixed. ---
    // grad_A[i, j] += d_inv_sqrt[i] * d_inv_sqrt[j] * grad_A_hat[i, j]  (for i != j)
    // Self-loops: grad_A[i, i] gets d_inv_sqrt[i] * d_inv_sqrt[i] * grad_A_hat[i, i],
    // but the original A has no self-loops (typically); for symmetry with the
    // input adj's diagonal (which is 0), we set grad_A's diagonal to the same.
    // We add the grad_A contribution from grad_A' (the upstream).
    Tensor d_inv_sqrt(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j) deg += last_A_(i, j) + (i == j ? 1.0 : 0.0);
        d_inv_sqrt(i, 0) = (deg > 0.0) ? 1.0 / std::sqrt(deg) : 0.0;
    }

    Tensor grad_A_full(N, N);
    grad_A_full.fill(0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            grad_A_full(i, j) = d_inv_sqrt(i, 0) * d_inv_sqrt(j, 0) * grad_A_hat(i, j);
        }
    }
    // Scatter grad_A' back to the full N×N grad_A via idx
    for (size_t i = 0; i < N_out; ++i) {
        for (size_t j = 0; j < N_out; ++j) {
            grad_A_full(last_idx_[i], last_idx_[j]) += grad_A(i, j);
        }
    }
    last_grad_A_ = grad_A_full;

    (void)learning_rate;  // caller handles update via update_weights(lr)
    return grad_input;
}

void SAGPool::update_weights(double learning_rate) {
    for (size_t i = 0; i < in_features_; ++i) {
        W_pool_(i, 0) -= learning_rate * grad_W_pool_(i, 0);
    }
}

void SAGPool::zero_grad() {
    grad_W_pool_.fill(0.0);
    last_grad_A_ = Tensor(0, 0);
}

std::vector<Tensor*> SAGPool::parameters() {
    return {&W_pool_};
}

std::vector<Tensor*> SAGPool::gradients() {
    return {&grad_W_pool_};
}

Tensor SAGPool::get_weights() const {
    return W_pool_;
}

Tensor SAGPool::get_gradients() const {
    return grad_W_pool_;
}

// ============================================================================
// SAGPoolHierarchical
// ============================================================================

SAGPoolHierarchical::SAGPoolHierarchical(size_t in_features, size_t hidden_features,
                                          size_t num_classes, double ratio, size_t num_layers)
    : in_features_(in_features), hidden_features_(hidden_features), num_classes_(num_classes),
      ratio_(ratio), num_layers_(num_layers),
      readout_(hidden_features, num_classes) {
    if (in_features == 0 || hidden_features == 0 || num_classes == 0)
        throw std::invalid_argument("SAGPoolHierarchical: dims must be > 0");
    if (ratio <= 0.0 || ratio > 1.0)
        throw std::invalid_argument("SAGPoolHierarchical: ratio must be in (0, 1]");
    if (num_layers == 0)
        throw std::invalid_argument("SAGPoolHierarchical: num_layers must be > 0");

    gcn_blocks_.reserve(num_layers_);
    pool_blocks_.reserve(num_layers_);
    for (size_t l = 0; l < num_layers_; ++l) {
        // First layer takes in_features; subsequent layers take hidden_features
        size_t in_f = (l == 0) ? in_features_ : hidden_features_;
        gcn_blocks_.emplace_back(in_f, hidden_features_);
        gcn_blocks_[l].W.init_weights("xavier");
        pool_blocks_.emplace_back(hidden_features_, ratio_);
    }
}

Tensor SAGPoolHierarchical::forward(const Tensor& /*input*/) {
    throw std::runtime_error("SAGPoolHierarchical: forward(input) without adj not supported; use forward_with_adj");
}

std::pair<Tensor, Tensor> SAGPoolHierarchical::forward_with_adj(const Tensor& input, const Tensor& adj) {
    size_t N = input.rows;
    if (adj.rows != N || adj.cols != N)
        throw std::invalid_argument("SAGPoolHierarchical: adj must be N x N");

    last_A_ = adj.clone();
    layer_inputs_.clear();
    layer_inputs_.push_back(input.clone());

    Tensor cur_X = input.clone();
    Tensor cur_A = adj.clone();

    for (size_t l = 0; l < num_layers_; ++l) {
        // First layer takes in_features; subsequent layers take hidden_features
        size_t out_f = hidden_features_;

        GCNBlock& gb = gcn_blocks_[l];
        gb.last_input = cur_X.clone();
        size_t Nl = cur_X.rows;

        // Compute A_hat
        Tensor A_plus_I(Nl, Nl);
        for (size_t i = 0; i < Nl; ++i)
            for (size_t j = 0; j < Nl; ++j)
                A_plus_I(i, j) = cur_A(i, j) + (i == j ? 1.0 : 0.0);
        Tensor d_inv_sqrt(Nl, 1);
        for (size_t i = 0; i < Nl; ++i) {
            double deg = 0.0;
            for (size_t j = 0; j < Nl; ++j) deg += A_plus_I(i, j);
            d_inv_sqrt(i, 0) = (deg > 0.0) ? 1.0 / std::sqrt(deg) : 0.0;
        }
        gb.last_A_hat = Tensor(Nl, Nl);
        for (size_t i = 0; i < Nl; ++i)
            for (size_t j = 0; j < Nl; ++j)
                gb.last_A_hat(i, j) = d_inv_sqrt(i, 0) * A_plus_I(i, j) * d_inv_sqrt(j, 0);

        // AX = A_hat @ X
        gb.last_AX = Tensor(Nl, cur_X.cols);
        for (size_t i = 0; i < Nl; ++i)
            for (size_t j = 0; j < cur_X.cols; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < Nl; ++k) s += gb.last_A_hat(i, k) * cur_X(k, j);
                gb.last_AX(i, j) = s;
            }

        // AW = AX @ W
        Tensor AW(Nl, out_f);
        const Tensor& W = gb.W.get_weights();
        for (size_t i = 0; i < Nl; ++i)
            for (size_t j = 0; j < out_f; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < cur_X.cols; ++k) s += gb.last_AX(i, k) * W(k, j);
                AW(i, j) = s;
            }

        // ReLU
        gb.relu_mask.resize(Nl);
        Tensor h(Nl, out_f);
        for (size_t i = 0; i < Nl; ++i) {
            gb.relu_mask[i].resize(out_f);
            for (size_t j = 0; j < out_f; ++j) {
                gb.relu_mask[i][j] = (AW(i, j) > 0.0) ? 1.0 : 0.0;
                h(i, j) = std::max(0.0, AW(i, j));
            }
        }
        gb.last_output = h.clone();
        layer_inputs_.push_back(h.clone());

        // Pool
        PoolBlock& pb = pool_blocks_[l];
        auto result = pb.pool.forward_with_adj(h, cur_A);
        pb.last_input = h.clone();
        pb.last_X_out = result.first.clone();
        pb.last_idx = pb.pool.last_idx();
        pb.last_Z = pb.pool.last_Z();
        pb.last_N = Nl;
        pb.last_N_out = result.first.rows;
        layer_inputs_.push_back(result.first.clone());

        cur_X = result.first;
        cur_A = result.second;
    }

    // Readout: sum-pool across remaining nodes
    Tensor readout_input(1, hidden_features_);
    for (size_t j = 0; j < hidden_features_; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < cur_X.rows; ++i) s += cur_X(i, j);
        readout_input(0, j) = s;
    }
    last_readout_input_ = readout_input.clone();

    // Classifier
    readout_.last_input = readout_input.clone();
    Tensor classifier_out = readout_input * readout_.weights.transpose();
    for (size_t j = 0; j < classifier_out.cols; ++j) {
        double bj = readout_.bias(0, j);
        for (size_t i = 0; i < classifier_out.rows; ++i)
            classifier_out(i, j) += bj;
    }
    last_classifier_out_ = classifier_out.clone();
    return {classifier_out, Tensor()};
}

Tensor SAGPoolHierarchical::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // grad_output: (1, num_classes) — gradient of loss w.r.t. classifier output
    Tensor grad_readout = grad_output.clone();

    // Dense classifier backward
    Tensor grad_w_c = grad_readout.transpose() * last_readout_input_;  // (num_classes, hidden)
    Tensor grad_b_c(1, grad_readout.cols);
    for (size_t j = 0; j < grad_readout.cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < grad_readout.rows; ++i) sum += grad_readout(i, j);
        grad_b_c(0, j) = sum;
    }
    readout_.grad_weights += grad_w_c;
    readout_.grad_bias += grad_b_c;
    Tensor grad_after_readout = grad_readout * readout_.weights;  // (1, hidden)
    // Expand (1, hidden) -> (cur_N, hidden) via sum (since readout was sum-pool)
    // The last cur_X had shape (last_pool_N_out, hidden).
    size_t last_pool_N = pool_blocks_[num_layers_ - 1].last_N_out;
    Tensor grad_pooled_out(last_pool_N, hidden_features_);
    grad_pooled_out.fill(0.0);
    for (size_t j = 0; j < hidden_features_; ++j)
        for (size_t i = 0; i < last_pool_N; ++i)
            grad_pooled_out(i, j) = grad_after_readout(0, j);

    // Now back-propagate through pool → gcn → pool → gcn → ... in reverse order
    Tensor grad_cur_X = grad_pooled_out;
    Tensor grad_cur_A;  // we don't propagate adj grads through the model for now
    Tensor grad_cur_A_dummy;  // placeholder

    for (size_t l = num_layers_; l > 0; --l) {
        size_t idx = l - 1;
        // Pool backward (grad_A passed as zeros — we don't propagate A grads)
        Tensor zero_grad_A(pool_blocks_[idx].last_N_out, pool_blocks_[idx].last_N_out);
        zero_grad_A.fill(0.0);
        Tensor grad_pre_pool = pool_blocks_[idx].pool.backward_with_adj_grads(
            grad_cur_X, zero_grad_A, 0.0);

        // GCN backward
        GCNBlock& gb = gcn_blocks_[idx];
        Tensor& h = gb.last_output;
        Tensor grad_h = grad_pre_pool.clone();
        // ReLU backward
        for (size_t i = 0; i < h.rows; ++i)
            for (size_t j = 0; j < h.cols; ++j)
                grad_h(i, j) *= gb.relu_mask[i][j];

        // AX @ W backward
        Tensor& W = gb.W.weights;
        Tensor grad_W_loc = grad_h.transpose() * gb.last_AX;  // (hidden, in)
        Tensor grad_b_loc(1, grad_h.cols);
        for (size_t j = 0; j < grad_h.cols; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < grad_h.rows; ++i) s += grad_h(i, j);
            grad_b_loc(0, j) = s;
        }
        gb.W.grad_weights += grad_W_loc;
        gb.W.grad_bias += grad_b_loc;

        // grad_AX = grad_h @ W^T
        Tensor grad_AX(grad_h.rows, gb.last_AX.cols);
        for (size_t i = 0; i < grad_h.rows; ++i)
            for (size_t j = 0; j < gb.last_AX.cols; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < grad_h.cols; ++k) s += grad_h(i, k) * W(k, j);
                grad_AX(i, j) = s;
            }

        // grad_X = A_hat^T @ grad_AX
        Tensor grad_X_input(grad_AX.rows, gb.last_AX.cols);
        for (size_t i = 0; i < grad_AX.rows; ++i)
            for (size_t j = 0; j < gb.last_AX.cols; ++j) {
                double s = 0.0;
                for (size_t k = 0; k < grad_AX.rows; ++k) s += gb.last_A_hat(k, i) * grad_AX(k, j);
                grad_X_input(i, j) = s;
            }

        grad_cur_X = grad_X_input;
    }

    (void)grad_cur_A; (void)grad_cur_A_dummy;
    return grad_cur_X;
}

void SAGPoolHierarchical::update_weights(double learning_rate) {
    for (auto& gb : gcn_blocks_) gb.W.update_weights(learning_rate);
    for (auto& pb : pool_blocks_) pb.pool.update_weights(learning_rate);
    readout_.update_weights(learning_rate);
}

void SAGPoolHierarchical::zero_grad() {
    for (auto& gb : gcn_blocks_) gb.W.zero_grad();
    for (auto& pb : pool_blocks_) pb.pool.zero_grad();
    readout_.zero_grad();
}

std::vector<Tensor*> SAGPoolHierarchical::parameters() {
    std::vector<Tensor*> p;
    for (auto& gb : gcn_blocks_) {
        auto gp = gb.W.parameters();
        p.insert(p.end(), gp.begin(), gp.end());
    }
    for (auto& pb : pool_blocks_) {
        auto pp = pb.pool.parameters();
        p.insert(p.end(), pp.begin(), pp.end());
    }
    auto rp = readout_.parameters();
    p.insert(p.end(), rp.begin(), rp.end());
    return p;
}

std::vector<Tensor*> SAGPoolHierarchical::gradients() {
    std::vector<Tensor*> g;
    for (auto& gb : gcn_blocks_) {
        auto gg = gb.W.gradients();
        g.insert(g.end(), gg.begin(), gg.end());
    }
    for (auto& pb : pool_blocks_) {
        auto pg = pb.pool.gradients();
        g.insert(g.end(), pg.begin(), pg.end());
    }
    auto rg = readout_.gradients();
    g.insert(g.end(), rg.begin(), rg.end());
    return g;
}

// ============================================================================
// SAGPoolGlobal
// ============================================================================

SAGPoolGlobal::SAGPoolGlobal(size_t in_features, size_t hidden_features,
                              size_t num_classes, double ratio)
    : in_features_(in_features), hidden_features_(hidden_features),
      num_classes_(num_classes), ratio_(ratio),
      W_gcn_(in_features, hidden_features), pool_(hidden_features, ratio, true),
      readout_(hidden_features, num_classes) {
    if (in_features == 0 || hidden_features == 0 || num_classes == 0)
        throw std::invalid_argument("SAGPoolGlobal: dims must be > 0");
    if (ratio <= 0.0 || ratio > 1.0)
        throw std::invalid_argument("SAGPoolGlobal: ratio must be in (0, 1]");
    W_gcn_.init_weights("xavier");
}

Tensor SAGPoolGlobal::forward(const Tensor& /*input*/) {
    throw std::runtime_error("SAGPoolGlobal: forward(input) without adj not supported; use forward_with_adj");
}

std::pair<Tensor, Tensor> SAGPoolGlobal::forward_with_adj(const Tensor& input, const Tensor& adj) {
    size_t N = input.rows;
    if (adj.rows != N || adj.cols != N)
        throw std::invalid_argument("SAGPoolGlobal: adj must be N x N");
    last_input_ = input.clone();
    last_A_ = adj.clone();

    // GCN: A_hat X W_gcn + 0 bias, then ReLU
    Tensor A_plus_I(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A_plus_I(i, j) = adj(i, j) + (i == j ? 1.0 : 0.0);
    Tensor d_inv_sqrt(N, 1);
    for (size_t i = 0; i < N; ++i) {
        double deg = 0.0;
        for (size_t j = 0; j < N; ++j) deg += A_plus_I(i, j);
        d_inv_sqrt(i, 0) = (deg > 0.0) ? 1.0 / std::sqrt(deg) : 0.0;
    }
    last_A_hat_ = Tensor(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            last_A_hat_(i, j) = d_inv_sqrt(i, 0) * A_plus_I(i, j) * d_inv_sqrt(j, 0);

    last_AX_ = Tensor(N, input.cols);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < input.cols; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < N; ++k) s += last_A_hat_(i, k) * input(k, j);
            last_AX_(i, j) = s;
        }

    Tensor AW(N, hidden_features_);
    const Tensor& W = W_gcn_.get_weights();
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < hidden_features_; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < input.cols; ++k) s += last_AX_(i, k) * W(k, j);
            AW(i, j) = s;
        }

    last_relu_mask_.resize(N);
    Tensor h(N, hidden_features_);
    for (size_t i = 0; i < N; ++i) {
        last_relu_mask_[i].resize(hidden_features_);
        for (size_t j = 0; j < hidden_features_; ++j) {
            last_relu_mask_[i][j] = (AW(i, j) > 0.0) ? 1.0 : 0.0;
            h(i, j) = std::max(0.0, AW(i, j));
        }
    }
    last_gcn_out_ = h.clone();

    // Pool
    auto result = pool_.forward_with_adj(h, adj);
    last_X_pooled_ = result.first.clone();
    last_idx_ = pool_.last_idx();

    // Readout
    Tensor readout_input(1, hidden_features_);
    for (size_t j = 0; j < hidden_features_; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < result.first.rows; ++i) s += result.first(i, j);
        readout_input(0, j) = s;
    }
    last_readout_input_ = readout_input.clone();

    readout_.last_input = readout_input.clone();
    Tensor classifier_out = readout_input * readout_.weights.transpose();
    for (size_t j = 0; j < classifier_out.cols; ++j) {
        double bj = readout_.bias(0, j);
        for (size_t i = 0; i < classifier_out.rows; ++i)
            classifier_out(i, j) += bj;
    }
    last_classifier_out_ = classifier_out.clone();
    return {classifier_out, Tensor()};
}

Tensor SAGPoolGlobal::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // Classifier backward
    Tensor grad_readout = grad_output.clone();
    Tensor grad_w_c = grad_readout.transpose() * last_readout_input_;
    Tensor grad_b_c(1, grad_readout.cols);
    for (size_t j = 0; j < grad_readout.cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < grad_readout.rows; ++i) sum += grad_readout(i, j);
        grad_b_c(0, j) = sum;
    }
    readout_.grad_weights += grad_w_c;
    readout_.grad_bias += grad_b_c;

    // grad_after_readout: (1, hidden)
    Tensor grad_after_readout = grad_readout * readout_.weights;

    // Expand (1, hidden) -> (N_out, hidden) via broadcast (sum over nodes)
    size_t N_out = last_X_pooled_.rows;
    Tensor grad_pooled(N_out, hidden_features_);
    grad_pooled.fill(0.0);
    for (size_t j = 0; j < hidden_features_; ++j)
        for (size_t i = 0; i < N_out; ++i)
            grad_pooled(i, j) = grad_after_readout(0, j);

    // Pool backward (with zero grad_A — we don't propagate adj grads)
    Tensor zero_grad_A(N_out, N_out);
    zero_grad_A.fill(0.0);
    Tensor grad_pre_pool = pool_.backward_with_adj_grads(grad_pooled, zero_grad_A, 0.0);

    // GCN backward
    Tensor grad_h = grad_pre_pool.clone();
    // ReLU backward
    size_t N = last_gcn_out_.rows;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < hidden_features_; ++j)
            grad_h(i, j) *= last_relu_mask_[i][j];

    // grad_W_gcn = grad_h^T @ last_AX
    Tensor grad_W_loc = grad_h.transpose() * last_AX_;
    Tensor grad_b_loc(1, grad_h.cols);
    for (size_t j = 0; j < grad_h.cols; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < grad_h.rows; ++i) s += grad_h(i, j);
        grad_b_loc(0, j) = s;
    }
    W_gcn_.grad_weights += grad_W_loc;
    W_gcn_.grad_bias += grad_b_loc;

    // grad_AX = grad_h @ W_gcn^T
    const Tensor& W = W_gcn_.weights;
    Tensor grad_AX(N, last_AX_.cols);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < last_AX_.cols; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < hidden_features_; ++k) s += grad_h(i, k) * W(k, j);
            grad_AX(i, j) = s;
        }

    // grad_X = A_hat^T @ grad_AX
    Tensor grad_X_input(N, last_AX_.cols);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < last_AX_.cols; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < N; ++k) s += last_A_hat_(k, i) * grad_AX(k, j);
            grad_X_input(i, j) = s;
        }
    return grad_X_input;
}

void SAGPoolGlobal::update_weights(double learning_rate) {
    W_gcn_.update_weights(learning_rate);
    pool_.update_weights(learning_rate);
    readout_.update_weights(learning_rate);
}

void SAGPoolGlobal::zero_grad() {
    W_gcn_.zero_grad();
    pool_.zero_grad();
    readout_.zero_grad();
}

std::vector<Tensor*> SAGPoolGlobal::parameters() {
    std::vector<Tensor*> p;
    auto gp = W_gcn_.parameters();
    p.insert(p.end(), gp.begin(), gp.end());
    auto pp = pool_.parameters();
    p.insert(p.end(), pp.begin(), pp.end());
    auto rp = readout_.parameters();
    p.insert(p.end(), rp.begin(), rp.end());
    return p;
}

std::vector<Tensor*> SAGPoolGlobal::gradients() {
    std::vector<Tensor*> g;
    auto gg = W_gcn_.gradients();
    g.insert(g.end(), gg.begin(), gg.end());
    auto pg = pool_.gradients();
    g.insert(g.end(), pg.begin(), pg.end());
    auto rg = readout_.gradients();
    g.insert(g.end(), rg.begin(), rg.end());
    return g;
}
