// ============================================================================
// Flow Matching — implementation
// ============================================================================
#include "flow_matching.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <iostream>

namespace {

// Helper: check tensor shape.
inline void check_shape_eq(const Tensor& t, size_t r, size_t c, const char* what) {
    if (t.rows != r || t.cols != c) {
        throw std::invalid_argument(std::string("FlowMatching: ") + what +
                                    " expected (" + std::to_string(r) + ", " +
                                    std::to_string(c) + ") but got (" +
                                    std::to_string(t.rows) + ", " +
                                    std::to_string(t.cols) + ")");
    }
}

// Helper: SiLU derivative given the pre-activation x.
inline double silu_deriv_from_x(double x) {
    // d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    if (std::abs(x) < 1e-30) return 0.5;
    double sig = 1.0 / (1.0 + std::exp(-x));
    return sig + x * sig * (1.0 - sig);
}

// Helper: stable sigmoid.
inline double sigmoid_stable(double x) {
    if (x >= 0) {
        double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        double e = std::exp(x);
        return e / (1.0 + e);
    }
}

// Helper: sample N(0, 1) via Box-Muller.
inline double sample_standard_normal(std::mt19937& rng) {
    static thread_local std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

// Helper: column-vector mean of a (rows, cols) tensor.
inline double col_mean(const Tensor& t, size_t col) {
    double s = 0;
    for (size_t r = 0; r < t.rows; ++r) s += t[r][col];
    return s / static_cast<double>(t.rows);
}

// Helper: matrix L2 norm squared between rows i and j.
inline double row_l2sq(const Tensor& a, size_t i, const Tensor& b, size_t j) {
    double s = 0;
    for (size_t k = 0; k < a.cols; ++k) {
        double d = a[i][k] - b[j][k];
        s += d * d;
    }
    return s;
}

}  // anonymous namespace

// ============================================================================
// GaussianMixture2D
// ============================================================================
GaussianMixture2D::GaussianMixture2D(int n_per_cluster, int dim, double scale,
                                     double separation, unsigned seed)
    : n_per_cluster_(n_per_cluster), dim_(dim), scale_(scale),
      separation_(separation), seed_(seed) {}

std::pair<Tensor, Tensor> GaussianMixture2D::sample_pair() {
    int N = 2 * n_per_cluster_;
    Tensor x0(N, dim_);
    Tensor x1(N, dim_);

    // Cluster centres (in first 2 dims; if dim > 2, the rest are zero-centred).
    double cx[2] = { -separation_ / 2.0, +separation_ / 2.0 };
    double cy[2] = { -separation_ / 2.0, +separation_ / 2.0 };

    std::mt19937 rng(seed_);

    for (int i = 0; i < N; ++i) {
        int cluster = (i < n_per_cluster_) ? 0 : 1;
        double mx = cx[cluster];
        double my = cy[cluster];
        for (int d = 0; d < dim_; ++d) {
            if (d == 0) {
                x0[i][d] = mx + scale_ * sample_standard_normal(rng);
                x1[i][d] = mx + scale_ * sample_standard_normal(rng);
            } else if (d == 1) {
                x0[i][d] = my + scale_ * sample_standard_normal(rng);
                x1[i][d] = my + scale_ * sample_standard_normal(rng);
            } else {
                x0[i][d] = scale_ * sample_standard_normal(rng);
                x1[i][d] = scale_ * sample_standard_normal(rng);
            }
        }
    }
    return {x0, x1};
}

// ============================================================================
// FMTimeEmbedding
// ============================================================================
FMTimeEmbedding::FMTimeEmbedding(int hidden_dim) : hidden_dim_(hidden_dim) {
    if (hidden_dim_ <= 0) {
        throw std::invalid_argument("FMTimeEmbedding: hidden_dim must be > 0");
    }
}

Tensor FMTimeEmbedding::forward(double t) const {
    Tensor out(1, hidden_dim_);
    int half = hidden_dim_ / 2;
    // Standard Vaswani sinusoid: e[2k] = cos(w_k * t), e[2k+1] = sin(w_k * t)
    // with w_k = 1 / 10000^(2k/d). For half-channel, k = 0, ..., half-1.
    // Convention: even channels are cos, odd channels are sin. (This matches
    // the typical "embedding(pos, 2i) = sin, embedding(pos, 2i+1) = cos" but
    // inverted — here we put cos first so that at t=0, channel 0 = cos(0) = 1
    // and channel 1 = sin(0) = 0.)
    if (half == 0) {
        // hidden_dim == 1: degenerate but valid; emit constant 0 (no freq available)
        out[0][0] = 0.0;
        return out;
    }
    for (int k = 0; k < half; ++k) {
        double denom = std::pow(10000.0, (2.0 * k) / std::max(1, half - 1));
        double w = (denom == 0.0) ? 1.0 : (1.0 / denom);
        if (2 * k < hidden_dim_)     out[0][2 * k]     = std::cos(w * t);
        if (2 * k + 1 < hidden_dim_) out[0][2 * k + 1] = std::sin(w * t);
    }
    // If hidden_dim is odd, fill the last slot with sin(0) = 0.
    if (hidden_dim_ % 2 == 1 && hidden_dim_ >= 1) {
        out[0][hidden_dim_ - 1] = 0.0;
    }
    return out;
}

// ============================================================================
// ClassEmbedding
// ============================================================================
ClassEmbedding::ClassEmbedding(int num_classes, int hidden_dim, unsigned seed)
    : num_classes_(num_classes), hidden_dim_(hidden_dim) {
    if (num_classes_ <= 0) {
        throw std::invalid_argument("ClassEmbedding: num_classes must be > 0");
    }
    if (hidden_dim_ <= 0) {
        throw std::invalid_argument("ClassEmbedding: hidden_dim must be > 0");
    }
    // Deterministic init via Tensor::random using a fixed scale.
    // Use a small scale (0.1) so the embeddings start near zero.
    embeddings_ = Tensor::random(static_cast<size_t>(num_classes_),
                                  static_cast<size_t>(hidden_dim_), 0.1);
    (void)seed;  // Tensor::random uses its own RNG; seed reserved for future use.
}

Tensor ClassEmbedding::forward(int label) const {
    if (label < 0 || label >= num_classes_) {
        throw std::invalid_argument("ClassEmbedding: label out of range");
    }
    Tensor out(1, hidden_dim_);
    for (int j = 0; j < hidden_dim_; ++j) {
        out[0][j] = embeddings_[label][j];
    }
    return out;
}

Tensor ClassEmbedding::forward(const Tensor& one_hot) const {
    if (one_hot.cols != static_cast<size_t>(num_classes_)) {
        throw std::invalid_argument("ClassEmbedding: one_hot.cols must equal num_classes");
    }
    Tensor out(one_hot.rows, hidden_dim_);
    // out[i][j] = sum_k one_hot[i][k] * embeddings_[k][j]
    for (size_t i = 0; i < one_hot.rows; ++i) {
        for (size_t j = 0; j < static_cast<size_t>(hidden_dim_); ++j) {
            double s = 0.0;
            for (size_t k = 0; k < one_hot.cols; ++k) {
                s += one_hot[i][k] * embeddings_[k][j];
            }
            out[i][j] = s;
        }
    }
    return out;
}

// ============================================================================
// FlowMatchingNet
// ============================================================================
FlowMatchingNet::FlowMatchingNet(int data_dim, int hidden_dim, int num_classes,
                                 unsigned seed)
    : data_dim_(data_dim), hidden_dim_(hidden_dim), num_classes_(num_classes) {
    if (data_dim_ <= 0) {
        throw std::invalid_argument("FlowMatchingNet: data_dim must be > 0");
    }
    if (hidden_dim_ <= 0) {
        throw std::invalid_argument("FlowMatchingNet: hidden_dim must be > 0");
    }
    if (num_classes_ < 0) {
        throw std::invalid_argument("FlowMatchingNet: num_classes must be >= 0");
    }
    input_dim_ = data_dim_ + 1 + num_classes_;

    dense1_ = std::make_unique<Dense>(static_cast<size_t>(input_dim_),
                                       static_cast<size_t>(hidden_dim_));
    dense2_ = std::make_unique<Dense>(static_cast<size_t>(hidden_dim_),
                                       static_cast<size_t>(data_dim_));
    (void)seed;
}

Tensor FlowMatchingNet::forward(const Tensor& input) {
    if (input.cols != static_cast<size_t>(input_dim_)) {
        throw std::invalid_argument("FlowMatchingNet: input.cols must equal input_dim");
    }
    last_input_ = input.clone();
    last_residual_ = Tensor(input.rows, static_cast<size_t>(data_dim_));
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < static_cast<size_t>(data_dim_); ++j) {
            last_residual_[i][j] = input[i][j];
        }
    }

    Tensor h1 = dense1_->forward(input);  // (N, hidden_dim)
    last_hidden_ = h1.clone();
    Tensor a1(h1.rows, h1.cols);
    for (size_t i = 0; i < h1.rows; ++i) {
        for (size_t j = 0; j < h1.cols; ++j) {
            double x = h1[i][j];
            // SiLU = x * sigmoid(x)
            a1[i][j] = x * sigmoid_stable(x);
        }
    }
    last_act_ = a1.clone();

    Tensor h2 = dense2_->forward(a1);  // (N, data_dim)
    // Residual: add the original input slice.
    Tensor out(h2.rows, h2.cols);
    for (size_t i = 0; i < h2.rows; ++i) {
        for (size_t j = 0; j < h2.cols; ++j) {
            out[i][j] = h2[i][j] + last_residual_[i][j];
        }
    }
    last_velocity_ = out.clone();
    return out;
}

Tensor FlowMatchingNet::backward(const Tensor& grad_output, double learning_rate) {
    // The forward was: out = h2 + residual;  h2 = dense2(a1);  a1 = silu(h1);  h1 = dense1(in)
    // So: d_loss/d_out = grad_output (call it g_out)
    //     d_loss/d_h2 = g_out  (residual is just an additive passthrough)
    //     d_loss/d_residual = g_out  (added to grad_input later)
    //     d_loss/d_a1 = dense2.backward(g_out, lr)
    //     d_loss/d_h1 = grad_a1 * silu'(h1)
    //     d_loss/d_in = dense1.backward(grad_h1, lr)
    // Then grad_input = grad_dense1 + grad_residual.

    Tensor g_out = grad_output;
    // grad_input accumulator for residual contribution
    Tensor grad_input(last_input_.rows, last_input_.cols);
    grad_input.fill(0.0);

    // Residual backward: gradient w.r.t. residual is g_out.
    // Residual = last_input[:, :data_dim], so grad_input[:, :data_dim] += g_out.
    for (size_t i = 0; i < grad_input.rows; ++i) {
        for (size_t j = 0; j < static_cast<size_t>(data_dim_); ++j) {
            grad_input[i][j] += g_out[i][j];
        }
    }

    // dense2.backward(g_out, lr) accumulates grad into dense2_->grad_*, and
    // returns grad_a1 (gradient w.r.t. its input a1, shape (N, hidden_dim)).
    Tensor grad_a1 = dense2_->backward(g_out, learning_rate);

    // SiLU backward: grad_h1 = grad_a1 * silu'(h1)
    Tensor grad_h1(grad_a1.rows, grad_a1.cols);
    for (size_t i = 0; i < grad_a1.rows; ++i) {
        for (size_t j = 0; j < grad_a1.cols; ++j) {
            double x = last_hidden_[i][j];
            double sig = sigmoid_stable(x);
            double dsilu = sig + x * sig * (1.0 - sig);
            grad_h1[i][j] = grad_a1[i][j] * dsilu;
        }
    }
    // dense1.backward(grad_h1, lr) accumulates grad into dense1_->grad_* and
    // returns grad_in (gradient w.r.t. its input, shape (N, input_dim)).
    Tensor grad_in = dense1_->backward(grad_h1, learning_rate);

    // Add the dense1 contribution to grad_input (residual contribution was added above).
    for (size_t i = 0; i < grad_input.rows; ++i) {
        for (size_t j = 0; j < grad_input.cols; ++j) {
            grad_input[i][j] += grad_in[i][j];
        }
    }

    // The forward() stored last_velocity_; update it to reflect the gradient
    // for sanity (not strictly needed).
    return grad_input;
}

void FlowMatchingNet::update_weights(double learning_rate) {
    dense1_->update_weights(learning_rate);
    dense2_->update_weights(learning_rate);
}

std::vector<Tensor*> FlowMatchingNet::parameters() {
    std::vector<Tensor*> p;
    auto p1 = dense1_->parameters();
    auto p2 = dense2_->parameters();
    p.insert(p.end(), p1.begin(), p1.end());
    p.insert(p.end(), p2.begin(), p2.end());
    return p;
}

std::vector<Tensor*> FlowMatchingNet::gradients() {
    std::vector<Tensor*> g;
    auto g1 = dense1_->gradients();
    auto g2 = dense2_->gradients();
    g.insert(g.end(), g1.begin(), g1.end());
    g.insert(g.end(), g2.begin(), g2.end());
    return g;
}

void FlowMatchingNet::zero_grad() {
    dense1_->zero_grad();
    dense2_->zero_grad();
}

Tensor FlowMatchingNet::get_weights() const {
    // Concatenate dense1 weights and dense2 weights along columns.
    Tensor w1 = dense1_->get_weights();
    Tensor w2 = dense2_->get_weights();
    Tensor out(w1.rows, w1.cols + w2.cols);
    for (size_t i = 0; i < w1.rows; ++i) {
        for (size_t j = 0; j < w1.cols; ++j) out[i][j] = w1[i][j];
        for (size_t j = 0; j < w2.cols; ++j) out[i][w1.cols + j] = w2[i][j];
    }
    return out;
}

Tensor FlowMatchingNet::get_gradients() const {
    Tensor g1 = dense1_->get_gradients();
    Tensor g2 = dense2_->get_gradients();
    Tensor out(g1.rows, g1.cols + g2.cols);
    for (size_t i = 0; i < g1.rows; ++i) {
        for (size_t j = 0; j < g1.cols; ++j) out[i][j] = g1[i][j];
        for (size_t j = 0; j < g2.cols; ++j) out[i][g1.cols + j] = g2[i][j];
    }
    return out;
}

// ============================================================================
// FlowMatching
// ============================================================================
FlowMatching::FlowMatching(int data_dim, int hidden_dim, int num_classes,
                            double sigma_min, bool use_ot, unsigned seed)
    : data_dim_(data_dim), hidden_dim_(hidden_dim), num_classes_(num_classes),
      sigma_min_(sigma_min), use_ot_(use_ot), seed_(seed) {
    if (data_dim_ <= 0) {
        throw std::invalid_argument("FlowMatching: data_dim must be > 0");
    }
    if (hidden_dim_ <= 0) {
        throw std::invalid_argument("FlowMatching: hidden_dim must be > 0");
    }
    if (num_classes_ < 0) {
        throw std::invalid_argument("FlowMatching: num_classes must be >= 0");
    }
    if (sigma_min_ < 0.0 || sigma_min_ >= 1.0) {
        throw std::invalid_argument("FlowMatching: sigma_min must be in [0, 1)");
    }
    net_ = std::make_unique<FlowMatchingNet>(data_dim_, hidden_dim_, num_classes_, seed_);
    rng_.seed(seed_ == 0 ? std::random_device{}() : seed_);
}

Tensor FlowMatching::interpolate(const Tensor& x0, const Tensor& x1,
                                  const Tensor& t_col) {
    // x_t = (1 - (1-σ)·t) * x0 + t * x1   (the conditional FM path; for σ=0 this is (1-t)x0 + t·x1).
    Tensor xt(x0.rows, x0.cols);
    for (size_t i = 0; i < x0.rows; ++i) {
        double t = t_col[i][0];
        double a = 1.0 - (1.0 - sigma_min_) * t;
        for (size_t j = 0; j < x0.cols; ++j) {
            xt[i][j] = a * x0[i][j] + t * x1[i][j];
        }
    }
    return xt;
}

Tensor FlowMatching::target_velocity(const Tensor& x0, const Tensor& x1) {
    // v_target = x1 - (1 - σ) * x0
    Tensor v(x0.rows, x0.cols);
    for (size_t i = 0; i < x0.rows; ++i) {
        for (size_t j = 0; j < x0.cols; ++j) {
            v[i][j] = x1[i][j] - (1.0 - sigma_min_) * x0[i][j];
        }
    }
    return v;
}

Tensor FlowMatching::greedy_ot_assign(const Tensor& x0, const Tensor& x1) {
    // For each x0[i], find the nearest x1[j] (squared L2), assign greedily.
    int N = static_cast<int>(x0.rows);
    Tensor x1_perm(N, x1.cols);

    std::vector<bool> taken(N, false);
    for (int i = 0; i < N; ++i) {
        double best_cost = std::numeric_limits<double>::infinity();
        int best_j = -1;
        for (int j = 0; j < N; ++j) {
            if (taken[j]) continue;
            double c = row_l2sq(x0, i, x1, j);
            if (c < best_cost) {
                best_cost = c;
                best_j = j;
            }
        }
        if (best_j < 0) {
            // Should never happen, but defend against underflow.
            best_j = i;
        }
        taken[best_j] = true;
        for (size_t k = 0; k < x1.cols; ++k) {
            x1_perm[i][k] = x1[best_j][k];
        }
    }
    return x1_perm;
}

Tensor FlowMatching::build_net_input(const Tensor& x_t, const Tensor& t_col) {
    int N = static_cast<int>(x_t.rows);
    int input_dim = data_dim_ + 1 + num_classes_;
    Tensor inp(N, input_dim);

    // If last_one_hot_ is empty (e.g. we never called sample() with class labels),
    // treat it as all-zeros so the net still sees a valid input.
    bool have_oh = (last_one_hot_.rows == static_cast<size_t>(N) &&
                    (num_classes_ == 0 || last_one_hot_.cols == static_cast<size_t>(num_classes_)));

    for (int i = 0; i < N; ++i) {
        int col = 0;
        for (int d = 0; d < data_dim_; ++d) {
            inp[i][col++] = x_t[i][d];
        }
        inp[i][col++] = t_col[i][0];
        if (num_classes_ > 0) {
            for (int k = 0; k < num_classes_; ++k) {
                inp[i][col++] = have_oh ? last_one_hot_[i][k] : 0.0;
            }
        }
    }
    return inp;
}

Tensor FlowMatching::one_hot(const std::vector<int>& labels, int num_classes) {
    Tensor oh(labels.size(), num_classes);
    oh.fill(0.0);
    for (size_t i = 0; i < labels.size(); ++i) {
        int l = labels[i];
        if (l < 0 || l >= num_classes) {
            throw std::invalid_argument("FlowMatching::one_hot: label out of range");
        }
        oh[i][l] = 1.0;
    }
    return oh;
}

Tensor FlowMatching::forward(const Tensor& x0, const Tensor& x1) {
    if (x0.rows != x1.rows || x0.cols != x1.cols) {
        throw std::invalid_argument("FlowMatching: x0 / x1 must have the same shape");
    }
    if (x0.cols != static_cast<size_t>(data_dim_)) {
        throw std::invalid_argument("FlowMatching: x0.cols must equal data_dim");
    }
    int N = static_cast<int>(x0.rows);

    // 1. Sample t ~ U(0, 1) per row.
    Tensor t_col(N, 1);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
        t_col[i][0] = u(rng_);
    }
    return forward_with_t(x0, x1, t_col);
}

Tensor FlowMatching::forward_with_t(const Tensor& x0, const Tensor& x1,
                                     const Tensor& t_col) {
    int N = static_cast<int>(x0.rows);

    last_t_vec_ = t_col.clone();

    // 2. Optionally permute x1 via greedy OT.
    Tensor x1_eff = x1;
    if (use_ot_) {
        x1_eff = greedy_ot_assign(x0, x1);
    }
    last_x1_perm_ = x1_eff.clone();

    // 3. Compute x_t and v_target.
    Tensor x_t = interpolate(x0, x1_eff, t_col);
    Tensor v_target = target_velocity(x0, x1_eff);
    last_x_t_ = x_t.clone();
    last_v_target_ = v_target.clone();

    // 4. Build net input and run forward.
    Tensor inp = build_net_input(x_t, t_col);
    Tensor v_pred = net_->forward(inp);
    last_v_pred_ = v_pred.clone();

    // 5. Loss = mean over batch of (1/N) * sum_d (v_pred - v_target)^2
    double total = 0.0;
    for (size_t i = 0; i < v_pred.rows; ++i) {
        for (size_t j = 0; j < v_pred.cols; ++j) {
            double d = v_pred[i][j] - v_target[i][j];
            total += d * d;
        }
    }
    double mean_loss = total / static_cast<double>(N * v_pred.cols);
    Tensor loss(1, 1);
    loss[0][0] = mean_loss;
    last_loss_ = loss;
    return loss;
}

Tensor FlowMatching::backward() {
    // d_loss / d_v_pred = (2 / (N*D)) * (v_pred - v_target)
    // If backward() is called before forward(), last_v_pred_ has zero rows
    // (default-constructed Tensor), so N=0 and we just return an empty gradient
    // without invoking the network backward.
    int N = static_cast<int>(last_v_pred_.rows);
    int D = static_cast<int>(last_v_pred_.cols);
    if (N == 0 || D == 0) {
        // Defensive: nothing to backprop. Caller should call forward() first.
        return Tensor(0, data_dim_ + 1 + num_classes_);
    }
    double scale = 2.0 / static_cast<double>(N * D);
    Tensor grad_v_pred(N, D);
    for (size_t i = 0; i < last_v_pred_.rows; ++i) {
        for (size_t j = 0; j < last_v_pred_.cols; ++j) {
            grad_v_pred[i][j] = scale * (last_v_pred_[i][j] - last_v_target_[i][j]);
        }
    }
    // Pass through net backward; lr=0 here because caller will do update_weights separately.
    Tensor grad_input = net_->backward(grad_v_pred, 0.0);
    return grad_input;
}

void FlowMatching::update_weights(double lr) {
    net_->update_weights(lr);
}

void FlowMatching::sample(int n_samples, int n_steps,
                           const std::vector<int>& class_labels, unsigned seed) {
    if (n_samples <= 0) {
        throw std::invalid_argument("FlowMatching::sample: n_samples must be > 0");
    }
    if (n_steps <= 0) {
        throw std::invalid_argument("FlowMatching::sample: n_steps must be > 0");
    }
    if (num_classes_ > 0) {
        if (static_cast<int>(class_labels.size()) != n_samples) {
            throw std::invalid_argument("FlowMatching::sample: class_labels.size() must equal n_samples");
        }
    }

    // 1. Initialize x_0 ~ N(0, I).
    std::mt19937 local_rng(seed == 0 ? std::random_device{}() : seed);
    Tensor x(n_samples, data_dim_);
    for (int i = 0; i < n_samples; ++i) {
        for (int d = 0; d < data_dim_; ++d) {
            x[i][d] = sample_standard_normal(local_rng);
        }
    }

    // 2. Optionally set the one-hot conditioning.
    Tensor saved_one_hot = last_one_hot_.clone();
    if (num_classes_ > 0) {
        last_one_hot_ = one_hot(class_labels, num_classes_);
    }

    // 3. Euler ODE from t=0 to t=1.
    double dt = 1.0 / static_cast<double>(n_steps);
    for (int step = 0; step < n_steps; ++step) {
        double t = static_cast<double>(step) / static_cast<double>(n_steps);
        // Build per-row t column.
        Tensor t_col(n_samples, 1);
        for (int i = 0; i < n_samples; ++i) t_col[i][0] = t;

        Tensor inp = build_net_input(x, t_col);
        Tensor v_pred = net_->forward(inp);

        // x_{k+1} = x_k + dt * v_pred
        for (int i = 0; i < n_samples; ++i) {
            for (int d = 0; d < data_dim_; ++d) {
                x[i][d] += dt * v_pred[i][d];
            }
        }
    }

    // Restore last_one_hot_.
    last_one_hot_ = saved_one_hot;
    last_samples_ = x.clone();
}

// ============================================================================
// ConditionalFlowMatching
// ============================================================================
ConditionalFlowMatching::ConditionalFlowMatching(int data_dim, int hidden_dim,
                                                  int num_classes, double sigma_min,
                                                  unsigned seed)
    : FlowMatching(data_dim, hidden_dim, num_classes, sigma_min, /*use_ot*/ false, seed) {
    if (sigma_min < 0.0 || sigma_min >= 1.0) {
        throw std::invalid_argument("ConditionalFlowMatching: sigma_min must be in [0, 1)");
    }
}

Tensor ConditionalFlowMatching::forward(const Tensor& x0, const Tensor& x1) {
    // The base interpolate()/target_velocity() already use sigma_min_, so this
    // is identical to FlowMatching::forward — we just override the API to make
    // the σ_min > 0 path explicit at the type level.
    return FlowMatching::forward(x0, x1);
}

// ============================================================================
// OptimalTransportFlowMatching
// ============================================================================
OptimalTransportFlowMatching::OptimalTransportFlowMatching(int data_dim, int hidden_dim,
                                                            int num_classes, double sigma_min,
                                                            unsigned seed)
    : FlowMatching(data_dim, hidden_dim, num_classes, sigma_min, /*use_ot*/ true, seed) {
    if (sigma_min < 0.0 || sigma_min >= 1.0) {
        throw std::invalid_argument("OptimalTransportFlowMatching: sigma_min must be in [0, 1)");
    }
}

Tensor OptimalTransportFlowMatching::forward(const Tensor& x0, const Tensor& x1) {
    // The base forward() already calls greedy_ot_assign when use_ot_ is true.
    return FlowMatching::forward(x0, x1);
}
