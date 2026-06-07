// ============================================================================
// Linformer — Wang et al. 2020 implementation
// ============================================================================
//
// Forward (one attention layer, single head, sequence-axis projection):
//
//   Q = input @ W_q                  # (n, d)
//   K = input @ W_k                  # (n, d)
//   V = input @ W_v                  # (n, d)
//   K_red = E @ K                    # (k, d)  =  E (k,n) @ K (n,d)
//   V_red = F @ V                    # (k, d)
//   scores = Q @ K_red^T / sqrt(d)   # (n, k)
//   A = row_softmax(scores)          # (n, k)
//   head_out = A @ V_red             # (n, d)
//   output = head_out @ W_o^T        # (n, d)  (note: W_o is stored as (d,d),
//                                              Dense convention is y=xW^T+b)
//
// We use a single head (no head_dim split) — this matches the simplest
// Linformer formulation. Multi-head Linformer would be a separate layer
// that runs several single-head Linformers in parallel.
//
// Backward is decomposed in reverse:
//   1) d_head_out from d_output via W_o
//   2) d_V_red, d_A from d_head_out
//   3) d_scores from d_A (via softmax backward)
//   4) d_Q, d_K_red from d_scores
//   5) d_K from d_K_red via E
//   6) d_V from d_V_red via F
//   7) d_W_q, d_W_k, d_W_v, d_W_o from d_Q, d_K, d_V, d_head_out
//   8) d_E, d_F from d_K_red, d_V_red  (only if learned_projection)
//   9) d_input = d_Q @ W_q^T + d_K @ W_k^T + d_V @ W_v^T
// ============================================================================

#include "linformer.h"
#include "../../activations/activations.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ----------------------------------------------------------------------------
// Helper: row-wise softmax over the last axis of a (rows, cols) Tensor.
// Returns a new tensor of the same shape with each row summing to 1.
// Uses the max-subtraction trick for numerical stability.
// ----------------------------------------------------------------------------
static Tensor row_softmax(const Tensor& x) {
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        // Find row max
        double row_max = x[i][0];
        for (size_t j = 1; j < x.cols; ++j) {
            if (x[i][j] > row_max) row_max = x[i][j];
        }
        // Compute exp(x - max) and sum
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            double e = std::exp(x[i][j] - row_max);
            result[i][j] = e;
            sum += e;
        }
        // Normalize
        double inv = 1.0 / (sum + 1e-12);
        for (size_t j = 0; j < x.cols; ++j) {
            result[i][j] *= inv;
        }
    }
    return result;
}

// ============================================================================
// LinformerAttention
// ============================================================================

LinformerAttention::LinformerAttention(size_t d_model, size_t seq_len,
                                       size_t proj_dim, bool learned_projection)
    : d_model_(d_model),
      seq_len_(seq_len),
      proj_dim_(proj_dim == 0 ? seq_len : proj_dim),
      learned_projection_(learned_projection),
      scale_(1.0 / std::sqrt(static_cast<double>(d_model)))
{
    if (proj_dim_ > seq_len_) {
        throw std::invalid_argument(
            "LinformerAttention: proj_dim cannot exceed seq_len");
    }

    // W_q, W_k, W_v, W_o: (d_model, d_model) — Dense convention y = xW^T + b
    // We initialize with small random (xavier-style).
    W_q  = Tensor::random(d_model_, d_model_, 0.02);
    W_k  = Tensor::random(d_model_, d_model_, 0.02);
    W_v  = Tensor::random(d_model_, d_model_, 0.02);
    W_o  = Tensor::random(d_model_, d_model_, 0.02);
    grad_W_q = Tensor::zeros(d_model_, d_model_);
    grad_W_k = Tensor::zeros(d_model_, d_model_);
    grad_W_v = Tensor::zeros(d_model_, d_model_);
    grad_W_o = Tensor::zeros(d_model_, d_model_);

    // E, F: (k, n) — applied as (k, n) @ (n, d) = (k, d)
    if (learned_projection_) {
        // Learned: init small random so projection starts as mild perturbation
        // around identity-like. We use xavier scale for (k, n) shape.
        double bound = std::sqrt(6.0 / static_cast<double>(proj_dim_ + seq_len_));
        std::mt19937 gen(2024);
        std::uniform_real_distribution<> dis(-bound, bound);
        E_ = Tensor(proj_dim_, seq_len_);
        F_ = Tensor(proj_dim_, seq_len_);
        for (size_t i = 0; i < proj_dim_; ++i) {
            for (size_t j = 0; j < seq_len_; ++j) {
                E_(i, j) = dis(gen);
                F_(i, j) = dis(gen);
            }
        }
        grad_E_ = Tensor::zeros(proj_dim_, seq_len_);
        grad_F_ = Tensor::zeros(proj_dim_, seq_len_);
    } else {
        // Fixed: each row i of E is a random unit vector of length n, with
        // different sign pattern. We use a Gaussian projection normalized to
        // unit length per row, then divide by sqrt(proj_dim) so that the
        // Johnson-Lindenstrauss-style concentration holds.
        // This is the "random Gaussian projection" variant of Linformer.
        std::mt19937 gen(2024);
        std::normal_distribution<double> ndis(0.0, 1.0);
        E_ = Tensor(proj_dim_, seq_len_);
        F_ = Tensor(proj_dim_, seq_len_);
        for (size_t i = 0; i < proj_dim_; ++i) {
            double sumsq = 0.0;
            for (size_t j = 0; j < seq_len_; ++j) {
                E_(i, j) = ndis(gen);
                F_(i, j) = ndis(gen);
                sumsq += E_(i, j) * E_(i, j);
            }
            // Normalize so ||E_i||_2 = 1 / sqrt(proj_dim)
            double scale = 1.0 / (std::sqrt(sumsq) * std::sqrt(static_cast<double>(proj_dim_)));
            double sumsq_F = 0.0;
            for (size_t j = 0; j < seq_len_; ++j) {
                E_(i, j) *= scale;
                sumsq_F += F_(i, j) * F_(i, j);
            }
            double scale_F = 1.0 / (std::sqrt(sumsq_F) * std::sqrt(static_cast<double>(proj_dim_)));
            for (size_t j = 0; j < seq_len_; ++j) {
                F_(i, j) *= scale_F;
            }
        }
        // grad buffers still exist but stay zero (we just zero them again to
        // be safe — they are unused when !learned_projection_).
        grad_E_ = Tensor::zeros(proj_dim_, seq_len_);
        grad_F_ = Tensor::zeros(proj_dim_, seq_len_);
    }
}

std::vector<Tensor*> LinformerAttention::parameters() {
    if (learned_projection_) {
        return { &W_q, &W_k, &W_v, &W_o, &E_, &F_ };
    }
    return { &W_q, &W_k, &W_v, &W_o };
}

std::vector<Tensor*> LinformerAttention::gradients() {
    if (learned_projection_) {
        return { &grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o, &grad_E_, &grad_F_ };
    }
    return { &grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o };
}

void LinformerAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_E_.fill(0.0);
    grad_F_.fill(0.0);
}

void LinformerAttention::update_weights(double lr) {
    auto sgd_update = [&](Tensor& W, const Tensor& gW) {
        for (size_t i = 0; i < W.rows; ++i)
            for (size_t j = 0; j < W.cols; ++j)
                W[i][j] -= lr * gW[i][j];
    };
    sgd_update(W_q, grad_W_q);
    sgd_update(W_k, grad_W_k);
    sgd_update(W_v, grad_W_v);
    sgd_update(W_o, grad_W_o);
    if (learned_projection_) {
        sgd_update(E_, grad_E_);
        sgd_update(F_, grad_F_);
    }
}

Tensor LinformerAttention::forward(const Tensor& input) {
    if (input.rows != seq_len_ || input.cols != d_model_) {
        throw std::invalid_argument(
            "LinformerAttention: input must be (seq_len, d_model)");
    }
    // Cache input
    last_input_ = input.clone();

    // Q, K, V projections
    // input is (n, d), W_q is (d, d), so Q = input @ W_q is (n, d).
    last_q_ = input * W_q;
    last_k_ = input * W_k;
    last_v_ = input * W_v;

    // Sequence-axis projection for K and V:
    //   K_red = E @ K       E: (k, n), K: (n, d)  ->  K_red: (k, d)
    //   V_red = F @ V
    last_k_reduced_ = E_ * last_k_;
    last_v_reduced_ = F_ * last_v_;

    // scores = Q @ K_red^T / sqrt(d)
    //   Q: (n, d), K_red^T: (d, k)  ->  scores: (n, k)
    Tensor scores = last_q_ * last_k_reduced_.transpose() * scale_;
    last_attn_ = row_softmax(scores);

    // head_out = A @ V_red     A: (n, k), V_red: (k, d)  ->  (n, d)
    last_attn_output_ = last_attn_ * last_v_reduced_;

    // output = head_out @ W_o^T
    //   head_out: (n, d), W_o: (d, d), W_o^T: (d, d)  ->  (n, d)
    last_output_ = last_attn_output_ * W_o.transpose();

    return last_output_;
}

Tensor LinformerAttention::backward(const Tensor& grad_output, double /*lr*/) {
    if (grad_output.rows != seq_len_ || grad_output.cols != d_model_) {
        throw std::invalid_argument(
            "LinformerAttention::backward: grad_output shape mismatch");
    }

    // 1) d_head_out from d_output via W_o
    //    output = head_out @ W_o^T  ->  d_head_out = d_output @ W_o
    //    dW_o    = d_output^T @ head_out    (d, n) @ (n, d) = (d, d)
    Tensor d_head_out = grad_output * W_o;
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t n = 0; n < seq_len_; ++n) s += grad_output[n][i] * last_attn_output_[n][j];
            grad_W_o[i][j] += s;
        }
    }

    // 2) d_V_red and d_A from d_head_out
    //    head_out = A @ V_red
    //    dV_red = A^T @ d_head_out        (k, n) @ (n, d) = (k, d)
    //    dA     = d_head_out @ V_red^T    (n, d) @ (d, k) = (n, k)
    Tensor d_v_red = last_attn_.transpose() * d_head_out;
    Tensor d_attn  = d_head_out * last_v_reduced_.transpose();

    // 3) d_scores from d_A (softmax backward)
    //    For a row softmax: dL/dz_i = sum_j ( dL/dA_j * A_j * (delta_ij - A_i) )
    //                      = A_i * ( dL/dA_i - sum_j dL/dA_j * A_j )
    Tensor d_scores(d_attn.rows, d_attn.cols);
    for (size_t i = 0; i < d_attn.rows; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < d_attn.cols; ++j) dot += d_attn[i][j] * last_attn_[i][j];
        for (size_t j = 0; j < d_attn.cols; ++j) {
            d_scores[i][j] = last_attn_[i][j] * (d_attn[i][j] - dot);
        }
    }
    // Apply 1/sqrt(d) scale (forward multiplied by scale_)
    for (size_t i = 0; i < d_scores.rows; ++i)
        for (size_t j = 0; j < d_scores.cols; ++j)
            d_scores[i][j] *= scale_;

    // 4) d_Q and d_K_red from d_scores
    //    scores = Q @ K_red^T
    //    dQ     = d_scores @ K_red        (n, k) @ (k, d) = (n, d)
    //    dK_red = d_scores^T @ Q          (k, n) @ (n, d) = (k, d)
    Tensor d_q     = d_scores * last_k_reduced_;
    Tensor d_k_red = d_scores.transpose() * last_q_;

    // 5) d_K from d_K_red via E (learned or fixed)
    //    K_red = E @ K    ->  dK = E^T @ dK_red   (n, k) @ (k, d) = (n, d)
    Tensor d_k = E_.transpose() * d_k_red;

    // 6) d_V from d_V_red via F
    Tensor d_v = F_.transpose() * d_v_red;

    // 7) d_W_q, d_W_k, d_W_v
    //    Q = input @ W_q  ->  dW_q = input^T @ dQ     (d, n) @ (n, d) = (d, d)
    //    K = input @ W_k  ->  dW_k = input^T @ dK
    //    V = input @ W_v  ->  dW_v = input^T @ dV
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double sq = 0.0, sk = 0.0, sv = 0.0;
            for (size_t n = 0; n < seq_len_; ++n) {
                sq += last_input_[n][i] * d_q[n][j];
                sk += last_input_[n][i] * d_k[n][j];
                sv += last_input_[n][i] * d_v[n][j];
            }
            grad_W_q[i][j] += sq;
            grad_W_k[i][j] += sk;
            grad_W_v[i][j] += sv;
        }
    }

    // 8) d_E, d_F from d_K_red, d_V_red (only if learned)
    //    K_red = E @ K  ->  dE = dK_red @ K^T    (k, d) @ (d, n) = (k, n)
    //    V_red = F @ V  ->  dF = dV_red @ V^T
    if (learned_projection_) {
        for (size_t i = 0; i < proj_dim_; ++i) {
            for (size_t j = 0; j < seq_len_; ++j) {
                double se = 0.0, sf = 0.0;
                for (size_t dd = 0; dd < d_model_; ++dd) {
                    se += d_k_red[i][dd] * last_k_[j][dd];
                    sf += d_v_red[i][dd] * last_v_[j][dd];
                }
                grad_E_[i][j] += se;
                grad_F_[i][j] += sf;
            }
        }
    }

    // 9) d_input = dQ @ W_q^T + dK @ W_k^T + dV @ W_v^T
    //
    //    Q = input @ W_q  with input (n, d), W_q (d, d).
    //    Q[n][j] = sum_i input[n][i] * W_q[i][j]
    //    d_input[n][i] = sum_j dQ[n][j] * W_q[i][j]
    //
    //    So d_input = dQ @ W_q^T (not dQ @ W_q).
    Tensor d_input(seq_len_, d_model_);
    for (size_t n = 0; n < seq_len_; ++n) {
        for (size_t i = 0; i < d_model_; ++i) {
            double s = 0.0;
            for (size_t kk = 0; kk < d_model_; ++kk) {
                s += d_q[n][kk] * W_q[i][kk];
                s += d_k[n][kk] * W_k[i][kk];
                s += d_v[n][kk] * W_v[i][kk];
            }
            d_input[n][i] = s;
        }
    }
    return d_input;
}

// ============================================================================
// LinformerBlock
// ============================================================================

LinformerBlock::LinformerBlock(size_t d_model, size_t seq_len, size_t proj_dim,
                               size_t ffn_dim, bool learned_projection)
    : d_model_(d_model),
      seq_len_(seq_len),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      ln1_(d_model),
      attn_(d_model, seq_len, proj_dim, learned_projection),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim_),    // (in=d_model, out=ffn_dim)  hidden expansion
      fcn_fc2_(ffn_dim_, d_model)     // (in=ffn_dim, out=d_model)  projection back
{}

std::vector<Tensor*> LinformerBlock::parameters() {
    auto v1 = ln1_.parameters();
    auto v2 = attn_.parameters();
    auto v3 = ln2_.parameters();
    auto v4 = ffn_fc1_.parameters();
    auto v5 = fcn_fc2_.parameters();
    v1.insert(v1.end(), v2.begin(), v2.end());
    v1.insert(v1.end(), v3.begin(), v3.end());
    v1.insert(v1.end(), v4.begin(), v4.end());
    v1.insert(v1.end(), v5.begin(), v5.end());
    return v1;
}

std::vector<Tensor*> LinformerBlock::gradients() {
    auto v1 = ln1_.gradients();
    auto v2 = attn_.gradients();
    auto v3 = ln2_.gradients();
    auto v4 = ffn_fc1_.gradients();
    auto v5 = fcn_fc2_.gradients();
    v1.insert(v1.end(), v2.begin(), v2.end());
    v1.insert(v1.end(), v3.begin(), v3.end());
    v1.insert(v1.end(), v4.begin(), v4.end());
    v1.insert(v1.end(), v5.begin(), v5.end());
    return v1;
}

void LinformerBlock::zero_grad() {
    ln1_.zero_grad();
    attn_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    fcn_fc2_.zero_grad();
}

void LinformerBlock::update_weights(double lr) {
    ln1_.update_weights(lr);
    attn_.update_weights(lr);
    ln2_.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    fcn_fc2_.update_weights(lr);
}

Tensor LinformerBlock::forward(const Tensor& input) {
    if (input.rows != seq_len_ || input.cols != d_model_) {
        throw std::invalid_argument(
            "LinformerBlock: input must be (seq_len, d_model)");
    }
    last_input_ = input.clone();

    // Pre-LN + attention + residual
    last_z1_       = ln1_.forward(input);
    last_attn_out_ = attn_.forward(last_z1_);
    last_res1_     = input + last_attn_out_;

    // Pre-LN + FFN + residual
    last_z2_           = ln2_.forward(last_res1_);
    Tensor ffn_h_pre   = ffn_fc1_.forward(last_z2_);
    last_ffn_hidden_   = ffn_h_pre.apply(GELU{});
    last_ffn_out_      = fcn_fc2_.forward(last_ffn_hidden_);
    return last_res1_ + last_ffn_out_;
}

Tensor LinformerBlock::backward(const Tensor& grad_output, double lr) {
    if (grad_output.rows != seq_len_ || grad_output.cols != d_model_) {
        throw std::invalid_argument(
            "LinformerBlock::backward: grad_output shape mismatch");
    }

    // d_res1 from grad_output (output = res1 + ffn_out -> d_res1 += grad_output)
    // d_ffn_out from grad_output (and d_ffn_out gets passed through fcn_fc2)
    Tensor d_res1    = grad_output;
    Tensor d_ffn_out = grad_output;

    // FFN backward: fcn_fc2 -> GELU -> ffn_fc1 -> ln2
    Tensor d_z2_post = fcn_fc2_.backward(d_ffn_out, lr);

    // We need pre-GELU activation. Re-run ffn_fc1 to recover it.
    // (We could cache last_ffn_pre_, but recomputation is the standard
    //  memory-saving trick in transformer training.)
    Tensor ffn_h_pre = ffn_fc1_.forward(last_z2_);

    // GELU backward
    Tensor d_ffn_h(d_z2_post.rows, d_z2_post.cols);
    for (size_t i = 0; i < d_z2_post.rows; ++i) {
        for (size_t j = 0; j < d_z2_post.cols; ++j) {
            double x = ffn_h_pre[i][j];
            // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            // GELU'(x) = 0.5 * (1 + erf(x/sqrt(2))) + x * phi(x)
            // where phi(x) = (1/sqrt(2*pi)) * exp(-x^2 / 2)
            double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
            double pdf = std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
            double gelu_deriv = cdf + x * pdf;
            d_ffn_h[i][j] = d_z2_post[i][j] * gelu_deriv;
        }
    }
    Tensor d_z2_pre = ffn_fc1_.backward(d_ffn_h, lr);

    // d_res1 accumulates BOTH the direct residual contribution from grad_output
    // AND the gradient flowing back through ln2 from the FFN path.
    d_res1 = d_res1 + ln2_.backward(d_z2_pre, lr);

    // Pre-LN attention path:
    //   res1 = x + attn(z1)  where z1 = ln1(x)
    //   d_z1 = attn_.backward(d_res1)         (grad w.r.t. attn's input = z1)
    //   d_x_via_ln1   = ln1_.backward(d_z1)   (x -> z1 via ln1)
    //   d_x_via_res   = d_res1                (x -> res1 via residual)
    //   d_x = d_x_via_ln1 + d_x_via_res
    Tensor d_z1 = attn_.backward(d_res1, lr);
    Tensor d_x_via_ln1 = ln1_.backward(d_z1, lr);

    Tensor d_x(d_x_via_ln1.rows, d_x_via_ln1.cols);
    for (size_t i = 0; i < d_x.rows; ++i)
        for (size_t j = 0; j < d_x.cols; ++j)
            d_x[i][j] = d_x_via_ln1[i][j] + d_res1[i][j];
    return d_x;
}

// ============================================================================
// LinformerModel
// ============================================================================

LinformerModel::LinformerModel(size_t d_model, size_t seq_len, size_t out_features,
                               size_t num_blocks, size_t proj_dim, size_t ffn_dim,
                               bool learned_projection)
    : d_model_(d_model), seq_len_(seq_len), out_features_(out_features),
      num_blocks_(num_blocks),
      blocks_(),
      classifier_(d_model, out_features)     // (in=d_model, out=out_features)
{
    for (size_t i = 0; i < num_blocks_; ++i) {
        blocks_.emplace_back(d_model, seq_len, proj_dim, ffn_dim, learned_projection);
    }
}

std::vector<Tensor*> LinformerModel::parameters() {
    std::vector<Tensor*> all;
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        all.insert(all.end(), bp.begin(), bp.end());
    }
    auto cp = classifier_.parameters();
    all.insert(all.end(), cp.begin(), cp.end());
    return all;
}

std::vector<Tensor*> LinformerModel::gradients() {
    std::vector<Tensor*> all;
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        all.insert(all.end(), bg.begin(), bg.end());
    }
    auto cg = classifier_.gradients();
    all.insert(all.end(), cg.begin(), cg.end());
    return all;
}

void LinformerModel::zero_grad() {
    for (auto& b : blocks_) b.zero_grad();
    classifier_.zero_grad();
}

void LinformerModel::update_weights(double lr) {
    for (auto& b : blocks_) b.update_weights(lr);
    classifier_.update_weights(lr);
}

Tensor LinformerModel::forward(const Tensor& input) {
    if (input.rows != seq_len_ || input.cols != d_model_) {
        throw std::invalid_argument(
            "LinformerModel: input must be (seq_len, d_model)");
    }
    last_input_ = input.clone();
    Tensor x = input;
    for (auto& b : blocks_) {
        x = b.forward(x);
    }
    // Classifier: (n, d) @ (d, out) -> (n, out)
    return classifier_.forward(x);
}

Tensor LinformerModel::backward(const Tensor& grad_output, double lr) {
    if (grad_output.rows != seq_len_ || grad_output.cols != out_features_) {
        throw std::invalid_argument(
            "LinformerModel::backward: grad_output shape mismatch");
    }
    // Classifier backward
    Tensor d_x = classifier_.backward(grad_output, lr);
    // Blocks in reverse
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d_x = it->backward(d_x, lr);
    }
    return d_x;
}
