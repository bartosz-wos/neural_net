// tabnet.cpp — TabNet: Hierarchical Attention for Tabular Deep Learning
//
// Reference: Arik & Pfister 2019 "TabNet: Attentive Interpretable Tabular
//   Learning" (https://arxiv.org/abs/1908.07442).
//
// Implementation notes
// --------------------
// * Forward, per decision step s = 0 .. num_steps-1:
//     1) masked[b, d]       = features[b, d] * mask[s][b, d]
//     2) h_indep[b, v]      = indep_block[s].forward(masked)         // (B, v)
//     3) h_shared[b, v]     = shared_encoder_.forward(h_indep)       // (B, v)
//     4) decision[b, c]     = ReLU(step_fc[s].forward(h_shared))      // (B, num_outputs)
//        output_accum       += decision
//     5) features := h_shared                                  (carry to next step)
//     6) scores[b, d]       = attention_block[s].w @ features[b, :] // (B, input_dim)
//     7) mask[s+1][b, d]    = softmax(prior_scale[s] * scores[b, d])
//     8) prior_scale[s+1]   = clamp(prior_scale[s] - gamma * (1 - mean_d mask[s][d]),
//                                   min=eps)
//   Step 0 starts from features = input, mask[0] = ones (no masking before first step).
// * Each "block" (shared_encoder_, indep_block[s]) is a 2-layer BN-FC-BN-FC-ReLU stack.
//   BatchNorm1D forward is implemented as: y = (x - mu_batch) / sqrt(var_batch + eps) * gamma + beta.
//   In training mode we use the running stats; we always operate in training mode for
//   simplicity (the dataset is small for tabular).
// * Sparse feature selection: softmax mask produces soft (non-sparse) feature gates.
//   Sparsemax (the canonical TabNet choice) is intentionally avoided for gradient-check
//   tractability — softmax is smoother and has a clean closed-form backward.
// * The decision contributions at all steps are accumulated into a single output, which
//   matches the canonical "aggregation across steps" of TabNet.
//
// Backward
// --------
// Implemented analytically through the full chain:
//   grad_output -> grad_step_fc -> grad_shared_encoder -> grad_indep_block -> grad_mask
//   -> grad_attention_w & grad_prior_scale -> grad_features (the residual h_shared)
//   -> propagated to the next step's grad_attention (since mask[s] is the residual mask
//   after step s).
//
// Parameter exposure
// ------------------
//   parameters() and gradients() return Tensor* lists in the canonical order:
//     shared_encoder_w1, b1, bn1_gamma, beta, w2, b2, bn2_gamma, beta,
//     indep_block[0..S-1]: w1, b1, bn1_gamma, beta, w2, b2, bn2_gamma, beta,
//     attention_block[0..S-1]: w,
//     step_fc[0..S-1]: w, b
//
// Conventions
// -----------
//   Dense convention: y = x W^T + b, W is (out, in), so dW[k, j] += sum_i d_pre[i, k] * x[i, j]
//                    dx = d_pre @ W
//   BatchNorm1D per-feature statistics over the batch axis.

#include "tabnet.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

// ============================================================================
// Small helpers (operate on raw Tensors so we don't need to thread BatchNorm1D
// Layer instances — the header stores BN params directly).
// ============================================================================

namespace {

// Dense forward: y[b, o] = sum_d x[b, d] * W[o, d] + b[0, o]
// W is (out_features, in_features); returns (batch, out_features).
Tensor dense_forward(const Tensor& x, const Tensor& W, const Tensor& b) {
    size_t B = x.rows;
    size_t D = x.cols;
    size_t O = W.rows;
    if (W.cols != D) throw std::invalid_argument("dense_forward: in dim mismatch");
    Tensor y(B, O);
    // y = x @ W^T -> (B, D) @ (D, O) = (B, O)
    for (size_t i = 0; i < B; ++i)
        for (size_t k = 0; k < D; ++k) {
            double xik = x(i, k);
            for (size_t j = 0; j < O; ++j)
                y(i, j) += xik * W(j, k);
        }
    for (size_t i = 0; i < B; ++i)
        for (size_t j = 0; j < O; ++j)
            y(i, j) += b(0, j);
    return y;
}

// Dense backward: returns (dx, dW, db) given upstream grad.
void dense_backward(const Tensor& x, const Tensor& W, const Tensor& d_pre,
                    Tensor& dx, Tensor& dW, Tensor& db) {
    size_t B = x.rows;
    size_t D = x.cols;
    size_t O = W.rows;
    if (W.cols != D) {
        std::cerr << "[dense_backward] W.cols != x.cols: W=" << W.rows << "x" << W.cols
                  << " x=" << B << "x" << D << std::endl;
        throw std::invalid_argument("dense_backward: W.cols != x.cols");
    }
    if (d_pre.rows != B || d_pre.cols != O) {
        std::cerr << "[dense_backward] d_pre mismatch: d_pre=" << d_pre.rows << "x" << d_pre.cols
                  << " expected " << B << "x" << O << std::endl;
        throw std::invalid_argument("dense_backward: d_pre shape mismatch");
    }
    dx = Tensor(B, D);
    // dx = d_pre @ W  -> (B, D)
    for (size_t i = 0; i < B; ++i)
        for (size_t j = 0; j < D; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < O; ++k) s += d_pre(i, k) * W(k, j);
            dx(i, j) = s;
        }
    // dW[k, j] = sum_i d_pre[i, k] * x[i, j]
    dW = Tensor(O, D);
    for (size_t k = 0; k < O; ++k)
        for (size_t j = 0; j < D; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < B; ++i) s += d_pre(i, k) * x(i, j);
            dW(k, j) = s;
        }
    // db[0, k] = sum_i d_pre[i, k]
    db = Tensor(1, O);
    for (size_t k = 0; k < O; ++k) {
        double s = 0.0;
        for (size_t i = 0; i < B; ++i) s += d_pre(i, k);
        db(0, k) = s;
    }
}

// BatchNorm1D forward (training mode, batch stats).  gamma, beta shape (1, F).
// Returns y and caches: mean[F], inv_std[F], x_centered[B, F], x_norm[B, F].
Tensor bn_forward(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                  double eps,
                  Tensor& mean_out, Tensor& inv_std_out,
                  Tensor& x_centered_out, Tensor& x_norm_out) {
    size_t B = x.rows;
    size_t F = x.cols;
    Tensor y(B, F);
    Tensor mean(1, F), inv_std(1, F);
    Tensor x_centered(B, F), x_norm(B, F);
    for (size_t f = 0; f < F; ++f) {
        double m = 0.0;
        for (size_t i = 0; i < B; ++i) m += x(i, f);
        m /= (double)B;
        mean(0, f) = m;
        double v = 0.0;
        for (size_t i = 0; i < B; ++i) {
            double d = x(i, f) - m;
            x_centered(i, f) = d;
            v += d * d;
        }
        v /= (double)B;
        double istd = 1.0 / std::sqrt(v + eps);
        inv_std(0, f) = istd;
        for (size_t i = 0; i < B; ++i) {
            double xn = (x(i, f) - m) * istd;
            x_norm(i, f) = xn;
            y(i, f) = gamma(0, f) * xn + beta(0, f);
        }
    }
    mean_out = mean;
    inv_std_out = inv_std;
    x_centered_out = x_centered;
    x_norm_out = x_norm;
    return y;
}

// BatchNorm1D backward: returns dx; computes dgamma, dbeta from upstream grad.
Tensor bn_backward(const Tensor& d_pre, const Tensor& x_centered, const Tensor& x_norm,
                   const Tensor& inv_std, const Tensor& gamma,
                   Tensor& dgamma, Tensor& dbeta) {
    size_t B = d_pre.rows;
    size_t F = d_pre.cols;
    dgamma = Tensor(1, F);
    dbeta = Tensor(1, F);
    Tensor dx(B, F);
    for (size_t f = 0; f < F; ++f) {
        double dg = 0.0, db_ = 0.0;
        for (size_t i = 0; i < B; ++i) {
            dg += d_pre(i, f) * x_norm(i, f);
            db_ += d_pre(i, f);
        }
        dgamma(0, f) = dg;
        dbeta(0, f) = db_;
        double g = gamma(0, f);
        // Standard BN backward (training-mode, batch stats).  See e.g. Ioffe & Szegedy 2015.
        double sum_dpre = 0.0;
        for (size_t i = 0; i < B; ++i) sum_dpre += d_pre(i, f);
        double sum_dpre_xc = 0.0;
        for (size_t i = 0; i < B; ++i) sum_dpre_xc += d_pre(i, f) * x_centered(i, f);
        double istd = inv_std(0, f);
        double istd3 = istd * istd * istd;
        for (size_t i = 0; i < B; ++i) {
            double dxhat = d_pre(i, f) * g;
            double xc = x_centered(i, f);
            dx(i, f) = istd * dxhat
                     - istd * sum_dpre / (double)B
                     - istd3 * xc * sum_dpre_xc / (double)B;
        }
    }
    return dx;
}

// ReLU forward (in place not — returns new tensor)
Tensor relu_forward(const Tensor& x) {
    Tensor y(x.rows, x.cols);
    for (size_t i = 0; i < x.data.size(); ++i) {
        y.data[i] = x.data[i] > 0.0 ? x.data[i] : 0.0;
    }
    return y;
}

// ReLU backward: returns elementwise mask * d_pre
Tensor relu_backward(const Tensor& x, const Tensor& d_pre) {
    Tensor dx(x.rows, x.cols);
    for (size_t i = 0; i < x.data.size(); ++i)
        dx.data[i] = (x.data[i] > 0.0) ? d_pre.data[i] : 0.0;
    return dx;
}

// Row-wise softmax.  Returns (rows, cols) with rows summing to 1.
__attribute__((unused))
static Tensor softmax_rows(const Tensor& x) {
    Tensor y(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double m = x(i, 0);
        for (size_t j = 1; j < x.cols; ++j) if (x(i, j) > m) m = x(i, j);
        double s = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            double e = std::exp(x(i, j) - m);
            y(i, j) = e;
            s += e;
        }
        double inv = 1.0 / s;
        for (size_t j = 0; j < x.cols; ++j) y(i, j) *= inv;
    }
    return y;
}

} // namespace

// ============================================================================
// TabNet constructor
// ============================================================================
TabNet::TabNet(int input_dim, int num_outputs, int num_steps,
               int num_attention_heads, int n_independent, double relaxation_factor)
    : input_dim_(input_dim), num_outputs_(num_outputs),
      num_steps_(num_steps < 1 ? 1 : num_steps),
      num_attention_heads_(num_attention_heads < 1 ? 1 : num_attention_heads),
      n_independent_(n_independent < 1 ? 1 : n_independent),
      relaxation_factor_(relaxation_factor),
      virtual_dim_(n_independent_ * 8),  // small hidden dim for the feature transformer
      training_(true)
{
    // Xavier-style init scale for FC layers.
    const double s = 0.1;

    auto init_block = [&](EncoderBlock& blk) {
        // EncoderBlock operates on h_indep which is (B, virtual_dim_).  So both
        // w1 and w2 are (virtual_dim_, virtual_dim_).
        blk.w1 = Tensor::random(virtual_dim_, virtual_dim_, s);
        blk.b1 = Tensor::zeros(1, virtual_dim_);
        blk.bn1_gamma = Tensor::zeros(1, virtual_dim_); blk.bn1_gamma.fill(1.0);
        blk.bn1_beta  = Tensor::zeros(1, virtual_dim_);
        blk.w2 = Tensor::random(virtual_dim_, virtual_dim_, s);
        blk.b2 = Tensor::zeros(1, virtual_dim_);
        blk.bn2_gamma = Tensor::zeros(1, virtual_dim_); blk.bn2_gamma.fill(1.0);
        blk.bn2_beta  = Tensor::zeros(1, virtual_dim_);
        blk.last_bn1_out            = Tensor::zeros(1, 1);
        blk.last_bn2_out            = Tensor::zeros(1, 1);
        blk.last_mean_bn1           = Tensor::zeros(1, 1);
        blk.last_inv_std_bn1        = Tensor::zeros(1, 1);
        blk.last_x_centered_bn1     = Tensor::zeros(1, 1);
        blk.last_x_norm_bn1         = Tensor::zeros(1, 1);
        blk.last_mean_bn2           = Tensor::zeros(1, 1);
        blk.last_inv_std_bn2        = Tensor::zeros(1, 1);
        blk.last_x_centered_bn2     = Tensor::zeros(1, 1);
        blk.last_x_norm_bn2         = Tensor::zeros(1, 1);
        blk.grad_bn1_gamma = Tensor::zeros(1, virtual_dim_);
        blk.grad_bn1_beta  = Tensor::zeros(1, virtual_dim_);
        blk.grad_bn2_gamma = Tensor::zeros(1, virtual_dim_);
        blk.grad_bn2_beta  = Tensor::zeros(1, virtual_dim_);
        blk.grad_w1 = Tensor::zeros(virtual_dim_, virtual_dim_);
        blk.grad_b1 = Tensor::zeros(1, virtual_dim_);
        blk.grad_w2 = Tensor::zeros(virtual_dim_, virtual_dim_);
        blk.grad_b2 = Tensor::zeros(1, virtual_dim_);
    };

    init_block(shared_encoder_);

    independent_blocks_.resize(num_steps_);
    for (auto& blk : independent_blocks_) {
        // IndependentBlock operates on masked input (B, virtual_dim_).  So both
        // w1 and w2 are (virtual_dim_, virtual_dim_).
        blk.w1 = Tensor::random(virtual_dim_, virtual_dim_, s);
        blk.b1 = Tensor::zeros(1, virtual_dim_);
        blk.bn1_gamma = Tensor::zeros(1, virtual_dim_); blk.bn1_gamma.fill(1.0);
        blk.bn1_beta  = Tensor::zeros(1, virtual_dim_);
        blk.w2 = Tensor::random(virtual_dim_, virtual_dim_, s);
        blk.b2 = Tensor::zeros(1, virtual_dim_);
        blk.bn2_gamma = Tensor::zeros(1, virtual_dim_); blk.bn2_gamma.fill(1.0);
        blk.bn2_beta  = Tensor::zeros(1, virtual_dim_);
        blk.last_bn1_out            = Tensor::zeros(1, 1);
        blk.last_bn2_out            = Tensor::zeros(1, 1);
        blk.last_mean_bn1           = Tensor::zeros(1, 1);
        blk.last_inv_std_bn1        = Tensor::zeros(1, 1);
        blk.last_x_centered_bn1     = Tensor::zeros(1, 1);
        blk.last_x_norm_bn1         = Tensor::zeros(1, 1);
        blk.last_mean_bn2           = Tensor::zeros(1, 1);
        blk.last_inv_std_bn2        = Tensor::zeros(1, 1);
        blk.last_x_centered_bn2     = Tensor::zeros(1, 1);
        blk.last_x_norm_bn2         = Tensor::zeros(1, 1);
        blk.grad_bn1_gamma = Tensor::zeros(1, virtual_dim_);
        blk.grad_bn1_beta  = Tensor::zeros(1, virtual_dim_);
        blk.grad_bn2_gamma = Tensor::zeros(1, virtual_dim_);
        blk.grad_bn2_beta  = Tensor::zeros(1, virtual_dim_);
        blk.grad_w1 = Tensor::zeros(virtual_dim_, virtual_dim_);
        blk.grad_b1 = Tensor::zeros(1, virtual_dim_);
        blk.grad_w2 = Tensor::zeros(virtual_dim_, virtual_dim_);
        blk.grad_b2 = Tensor::zeros(1, virtual_dim_);
    }

    attention_blocks_.resize(num_steps_);
    for (auto& ab : attention_blocks_) {
        // attention_w shape: (1, virtual_dim_) -> per-row scores of shape (B, 1) then
        // broadcast to (B, input_dim).  We store a (1, virtual_dim_) tensor to be
        // applied as scores[b, :] = features[b, :] @ attention_w^T (but we only need
        // a scalar per sample -> easier: w has shape (1, virtual_dim_), pre-activations
        // = features @ w^T (B, virtual_dim_) @ (virtual_dim_, 1) = (B, 1).
        ab.w = Tensor::random(1, virtual_dim_, s);
        ab.last_scores = Tensor::zeros(1, 1);
        ab.last_mask = Tensor::zeros(1, 1);
        ab.grad_w = Tensor::zeros(1, virtual_dim_);
    }

    step_fcs_.resize(num_steps_);
    for (auto& sf : step_fcs_) {
        sf.w = Tensor::random(num_outputs_, virtual_dim_, s);
        sf.b = Tensor::zeros(1, num_outputs_);
        sf.grad_w = Tensor::zeros(num_outputs_, virtual_dim_);
        sf.grad_b = Tensor::zeros(1, num_outputs_);
        sf.last_h_shared = Tensor::zeros(1, 1);
        sf.last_dec_pre = Tensor::zeros(1, 1);
    }

    // Input projection: maps input_dim -> virtual_dim.  Applied at step 0 to lift
    // the input into virtual_dim space so all subsequent step operations are uniform.
    W_in_  = Tensor::random(virtual_dim_, input_dim_, s);
    b_in_  = Tensor::zeros(1, virtual_dim_);
    grad_W_in_ = Tensor::zeros(virtual_dim_, input_dim_);
    grad_b_in_ = Tensor::zeros(1, virtual_dim_);
    last_input_proj_ = Tensor::zeros(1, 1);
    last_input_ = Tensor::zeros(1, 1);

    // Prior scales — scalar per step, initialized to 1.0.  Updated each forward.
    prior_scales_.resize(num_steps_);
    for (auto& p : prior_scales_) p = Tensor::zeros(1, 1);

    last_h_.resize(num_steps_);
    last_h_indep_.resize(num_steps_);
    last_masked_h_.resize(num_steps_);
    last_masks_.resize(num_steps_ + 1);   // initial mask + one per step

    // Build the parameter list now (and again as needed if size is queried first).
    auto refresh_param_list = [&]() {
        param_list_.clear();
        grad_list_.clear();
        // Shared encoder
        param_list_.push_back(&shared_encoder_.w1);
        param_list_.push_back(&shared_encoder_.b1);
        param_list_.push_back(&shared_encoder_.bn1_gamma);
        param_list_.push_back(&shared_encoder_.bn1_beta);
        param_list_.push_back(&shared_encoder_.w2);
        param_list_.push_back(&shared_encoder_.b2);
        param_list_.push_back(&shared_encoder_.bn2_gamma);
        param_list_.push_back(&shared_encoder_.bn2_beta);
        grad_list_.push_back(&shared_encoder_.grad_w1);
        grad_list_.push_back(&shared_encoder_.grad_b1);
        grad_list_.push_back(&shared_encoder_.grad_bn1_gamma);
        grad_list_.push_back(&shared_encoder_.grad_bn1_beta);
        grad_list_.push_back(&shared_encoder_.grad_w2);
        grad_list_.push_back(&shared_encoder_.grad_b2);
        grad_list_.push_back(&shared_encoder_.grad_bn2_gamma);
        grad_list_.push_back(&shared_encoder_.grad_bn2_beta);
        // Independent blocks
        for (auto& blk : independent_blocks_) {
            param_list_.push_back(&blk.w1);
            param_list_.push_back(&blk.b1);
            param_list_.push_back(&blk.bn1_gamma);
            param_list_.push_back(&blk.bn1_beta);
            param_list_.push_back(&blk.w2);
            param_list_.push_back(&blk.b2);
            param_list_.push_back(&blk.bn2_gamma);
            param_list_.push_back(&blk.bn2_beta);
            grad_list_.push_back(&blk.grad_w1);
            grad_list_.push_back(&blk.grad_b1);
            grad_list_.push_back(&blk.grad_bn1_gamma);
            grad_list_.push_back(&blk.grad_bn1_beta);
            grad_list_.push_back(&blk.grad_w2);
            grad_list_.push_back(&blk.grad_b2);
            grad_list_.push_back(&blk.grad_bn2_gamma);
            grad_list_.push_back(&blk.grad_bn2_beta);
        }
        // Attention blocks
        for (auto& ab : attention_blocks_) {
            param_list_.push_back(&ab.w);
            grad_list_.push_back(&ab.grad_w);
        }
        // Step FCs
        for (auto& sf : step_fcs_) {
            param_list_.push_back(&sf.w);
            param_list_.push_back(&sf.b);
            grad_list_.push_back(&sf.grad_w);
            grad_list_.push_back(&sf.grad_b);
        }
        // Input projection
        param_list_.push_back(&W_in_);
        param_list_.push_back(&b_in_);
        grad_list_.push_back(&grad_W_in_);
        grad_list_.push_back(&grad_b_in_);
    };
    refresh_param_list();
    (void)refresh_param_list;
}

// ============================================================================
// Forward
// ============================================================================
Tensor TabNet::forward(const Tensor& input) {
    const size_t B = input.rows;
    const size_t D = input.cols;
    if ((int)D != input_dim_)
        throw std::invalid_argument("TabNet::forward: input dim mismatch");
    if (B == 0)
        throw std::invalid_argument("TabNet::forward: empty batch");

    const size_t V = (size_t)virtual_dim_;
    const double eps = 1e-5;

    Tensor output(B, num_outputs_);
    output.fill(0.0);

    // Initial mask: ones over virtual_dim (no masking before step 0).  Stored as last_masks_[0].
    Tensor& cur_mask = last_masks_[0];
    cur_mask = Tensor(B, V);
    cur_mask.fill(1.0);

    double prior_scale = 1.0;

    // Step 0: project input to virtual_dim.
    last_input_ = input.clone();
    Tensor features = dense_forward(input, W_in_, b_in_);
    last_input_proj_ = features;

    for (int s = 0; s < num_steps_; ++s) {
        last_h_[s] = features;

        // ---- masked input (B, virtual_dim) ----
        Tensor masked(B, V);
        for (size_t i = 0; i < B; ++i)
            for (size_t j = 0; j < V; ++j)
                masked(i, j) = features(i, j) * cur_mask(i, j);
        last_masked_h_[s] = masked;

        // ---- indep block: bn1 -> fc1 -> bn2 -> fc2 -> relu ----
        IndependentBlock& indep = independent_blocks_[s];
        // FC1: virtual_dim -> virtual_dim
        Tensor fc1_pre = dense_forward(masked, indep.w1, indep.b1);
        // BN1
        Tensor bn1_y = bn_forward(fc1_pre, indep.bn1_gamma, indep.bn1_beta, eps,
                                   indep.last_mean_bn1, indep.last_inv_std_bn1,
                                   indep.last_x_centered_bn1, indep.last_x_norm_bn1);
        indep.last_bn1_out = bn1_y;
        // FC2: virtual_dim -> virtual_dim
        Tensor fc2_pre = dense_forward(bn1_y, indep.w2, indep.b2);
        // BN2
        Tensor bn2_y = bn_forward(fc2_pre, indep.bn2_gamma, indep.bn2_beta, eps,
                                   indep.last_mean_bn2, indep.last_inv_std_bn2,
                                   indep.last_x_centered_bn2, indep.last_x_norm_bn2);
        indep.last_bn2_out = bn2_y;
        // ReLU
        Tensor h_indep = relu_forward(bn2_y);
        last_h_indep_[s] = h_indep;

        // ---- shared encoder: same chain (operates on h_indep, virtual_dim) ----
        EncoderBlock& sh = shared_encoder_;
        Tensor sh_fc1 = dense_forward(h_indep, sh.w1, sh.b1);
        Tensor sh_bn1 = bn_forward(sh_fc1, sh.bn1_gamma, sh.bn1_beta, eps,
                                    sh.last_mean_bn1, sh.last_inv_std_bn1,
                                    sh.last_x_centered_bn1, sh.last_x_norm_bn1);
        sh.last_bn1_out = sh_bn1;
        Tensor sh_fc2 = dense_forward(sh_bn1, sh.w2, sh.b2);
        Tensor sh_bn2 = bn_forward(sh_fc2, sh.bn2_gamma, sh.bn2_beta, eps,
                                    sh.last_mean_bn2, sh.last_inv_std_bn2,
                                    sh.last_x_centered_bn2, sh.last_x_norm_bn2);
        sh.last_bn2_out = sh_bn2;
        Tensor h_shared = relu_forward(sh_bn2);

        // ---- decision contribution ----
        StepFC& sf = step_fcs_[s];
        Tensor dec_pre = dense_forward(h_shared, sf.w, sf.b);
        Tensor decision = relu_forward(dec_pre);
        // cache for backward
        sf.last_dec_pre = dec_pre;
        sf.last_h_shared = h_shared;
        // accumulate into output
        for (size_t i = 0; i < B; ++i)
            for (size_t j = 0; j < (size_t)num_outputs_; ++j)
                output(i, j) += decision(i, j);

        // ---- carry features ----
        features = h_shared;

        // ---- compute next mask ----
        AttentionBlock& ab = attention_blocks_[s];
        // softmax over virtual_dim: scores[b, v] = features[b, v] * ab.w[0, v]
        Tensor& next_mask = last_masks_[s + 1];
        next_mask = Tensor(B, V);
        // First compute and cache the pre-softmax scores (logits) so backward can
        // use them directly.
        Tensor& logits = ab.last_scores;
        logits = Tensor(B, V);
        for (size_t i = 0; i < B; ++i) {
            for (size_t j = 0; j < V; ++j)
                logits(i, j) = prior_scale * features(i, j) * ab.w(0, j);
            // softmax over V
            double mx = logits(i, 0);
            for (size_t j = 1; j < V; ++j) if (logits(i, j) > mx) mx = logits(i, j);
            double sm = 0.0;
            for (size_t j = 0; j < V; ++j) {
                double e = std::exp(logits(i, j) - mx);
                next_mask(i, j) = e;
                sm += e;
            }
            double inv = 1.0 / sm;
            for (size_t j = 0; j < V; ++j) next_mask(i, j) *= inv;
        }
        ab.last_mask = next_mask.clone();

        // ---- update prior scale: simple relaxation ----
        // mean mask usage at this step
        double mean_mask = 0.0;
        for (size_t i = 0; i < B; ++i)
            for (size_t j = 0; j < V; ++j)
                mean_mask += cur_mask(i, j);
        mean_mask /= (double)(B * V);
        prior_scale = std::max(0.0, prior_scale - relaxation_factor_ * (1.0 - mean_mask));
        // store per-step prior scale value as a 1x1 tensor (for completeness)
        prior_scales_[s](0, 0) = prior_scale;

        cur_mask = next_mask;
    }

    return output;
}

// ============================================================================
// Backward
// --------
Tensor TabNet::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t B = grad_output.rows;
    const size_t V = (size_t)virtual_dim_;
    // BN-internal eps is hard-coded inside bn_forward/bn_backward helpers.

    // 1. Each step's decision contribution receives the same grad_output (because output
    //    is a SUM across steps).  Compute per-step (grad_w, grad_b, grad_h_shared) and
    //    accumulate.
    std::vector<Tensor> step_grad_h_shared(num_steps_, Tensor(B, V));
    for (int s = 0; s < num_steps_; ++s) step_grad_h_shared[s].fill(0.0);

    for (int s = 0; s < num_steps_; ++s) {
        StepFC& sf = step_fcs_[s];
        Tensor d_dec = relu_backward(sf.last_dec_pre, grad_output);
        Tensor dx_in, dW, db;
        dense_backward(sf.last_h_shared, sf.w, d_dec, dx_in, dW, db);
        sf.grad_w += dW;
        sf.grad_b += db;
        // dx_in is the grad on h_shared at this step -> feed to shared_encoder backward.
        step_grad_h_shared[s] += dx_in;
    }

    // 2. Backward through the per-step chain (shared_encoder -> indep_block -> mask).
    //    features for the next step come from h_shared at step s.  So the grad on
    //    h_shared at step s also contributes via the "next-step mask" path.
    //
    //    We walk backward s = num_steps_-1 .. 0.  For each step s:
    //      total_grad_h_shared = step_grad_h_shared[s] + (contribution from mask[s+1]
    //                                                         backward through ab.w &
    //                                                         ab.last_scores softmax)
    Tensor grad_input(B, (size_t)input_dim_);
    grad_input.fill(0.0);

    for (int s = num_steps_ - 1; s >= 0; --s) {
        // ---- shared encoder backward ----
        EncoderBlock& sh = shared_encoder_;
        Tensor d_pre_relu = relu_backward(sh.last_bn2_out, step_grad_h_shared[s]);
        Tensor d_bn2 = bn_backward(d_pre_relu, sh.last_x_centered_bn2, sh.last_x_norm_bn2,
                                    sh.last_inv_std_bn2, sh.bn2_gamma,
                                    sh.grad_bn2_gamma, sh.grad_bn2_beta);
        Tensor dx_after_fc2, dW2, db2;
        dense_backward(sh.last_bn1_out, sh.w2, d_bn2, dx_after_fc2, dW2, db2);
        sh.grad_w2 += dW2;
        sh.grad_b2 += db2;
        Tensor d_bn1 = bn_backward(dx_after_fc2, sh.last_x_centered_bn1, sh.last_x_norm_bn1,
                                    sh.last_inv_std_bn1, sh.bn1_gamma,
                                    sh.grad_bn1_gamma, sh.grad_bn1_beta);
        Tensor dx_h_indep, dW1, db1;
        dense_backward(last_h_indep_[s], sh.w1, d_bn1, dx_h_indep, dW1, db1);
        sh.grad_w1 += dW1;
        sh.grad_b1 += db1;

        // dx_h_indep is the grad on h_indep at step s.  This flows back through the
        // ReLU of indep_block, then through indep_block to grad_masked_input.

        // ---- indep block backward ----
        IndependentBlock& indep = independent_blocks_[s];
        Tensor d_pre_relu_indep = relu_backward(indep.last_bn2_out, dx_h_indep);
        Tensor d_bn2_indep = bn_backward(d_pre_relu_indep,
                                           indep.last_x_centered_bn2, indep.last_x_norm_bn2,
                                           indep.last_inv_std_bn2, indep.bn2_gamma,
                                           indep.grad_bn2_gamma, indep.grad_bn2_beta);
        Tensor dx_after_fc2_indep, dW2i, db2i;
        dense_backward(indep.last_bn1_out, indep.w2, d_bn2_indep, dx_after_fc2_indep, dW2i, db2i);
        indep.grad_w2 += dW2i;
        indep.grad_b2 += db2i;
        Tensor d_bn1_indep = bn_backward(dx_after_fc2_indep,
                                           indep.last_x_centered_bn1, indep.last_x_norm_bn1,
                                           indep.last_inv_std_bn1, indep.bn1_gamma,
                                           indep.grad_bn1_gamma, indep.grad_bn1_beta);
        Tensor dx_masked, dW1i, db1i;
        dense_backward(last_masked_h_[s], indep.w1, d_bn1_indep, dx_masked, dW1i, db1i);
        indep.grad_w1 += dW1i;
        indep.grad_b1 += db1i;

        // dx_masked is the grad on masked_input at step s.  Flow back through the mask:
        //   masked_input[b, v] = features_prev[b, v] * mask[s][b, v]
        // So:
        //   grad features_prev[b, v] += dx_masked[b, v] * mask[s][b, v]
        //   grad mask[s][b, v]       += dx_masked[b, v] * features_prev[b, v]
        Tensor& msk = last_masks_[s];
        Tensor grad_features_prev(B, V);
        Tensor grad_mask(B, V);
        for (size_t i = 0; i < B; ++i)
            for (size_t v = 0; v < V; ++v) {
                grad_features_prev(i, v) = dx_masked(i, v) * msk(i, v);
                grad_mask(i, v) = dx_masked(i, v) * last_h_[s](i, v);
            }

        // ---- mask backward ----
        // mask[s] (for s > 0) was computed by softmax over logits = features_prev * ab.w[0, :].
        // mask[0] is ones (no grad).
        if (s > 0) {
            // The mask at step s was generated using attention_block[s-1] (during the
            // forward loop at step s-1, we computed mask[s] using ab[s-1]).
            AttentionBlock& prev_ab = attention_blocks_[s - 1];
            // Standard softmax backward:
            //   d_logits[b, v] = mask[b, v] * (grad_mask[b, v] - sum_v' grad_mask[b, v'] * mask[b, v'])
            Tensor& m = last_masks_[s];   // post-softmax mask
            Tensor d_logits(B, V);
            for (size_t i = 0; i < B; ++i) {
                double sm = 0.0;
                for (size_t v = 0; v < V; ++v) sm += grad_mask(i, v) * m(i, v);
                for (size_t v = 0; v < V; ++v)
                    d_logits(i, v) = m(i, v) * (grad_mask(i, v) - sm);
            }
            // logits[b, v] = prior_scale_{s-1} * features_prev[b, v] * prev_ab.w[0, v]
            // d_logits/b features_prev = prior_scale * prev_ab.w[0, v]
            // d_logits/b w[0, v]       = prior_scale * features_prev[b, v]
            double prev_ps = prior_scales_[s - 1](0, 0);
            Tensor& features_prev = last_h_[s];   // = h_shared[s-1]
            for (size_t v = 0; v < V; ++v) {
                double d_w = 0.0;
                for (size_t i = 0; i < B; ++i)
                    d_w += d_logits(i, v) * prev_ps * features_prev(i, v);
                prev_ab.grad_w(0, v) += d_w;
            }
            // grad features_prev += d_logits * prior_scale * prev_ab.w[0, v]
            for (size_t i = 0; i < B; ++i) {
                for (size_t v = 0; v < V; ++v) {
                    grad_features_prev(i, v) += d_logits(i, v) * prev_ps * prev_ab.w(0, v);
                }
            }
            // Add to step_grad_h_shared[s-1] (since h_shared[s-1] == features entering step s).
            for (size_t i = 0; i < B; ++i)
                for (size_t v = 0; v < V; ++v)
                    step_grad_h_shared[s - 1](i, v) += grad_features_prev(i, v);
        } else {
            // s == 0: features_prev is the input projection output.  Backward through
            // the input projection: y = last_input_ @ W_in_^T + b_in_ -> grad_features_prev.
            // d_W_in_(v, d) += sum_i grad_features_prev[i, v] * last_input_(i, d)
            // d_b_in_(0, v) += sum_i grad_features_prev[i, v]
            // d_input(i, d) = sum_v grad_features_prev[i, v] * W_in_(v, d)
            for (size_t v = 0; v < V; ++v) {
                double d_b = 0.0;
                for (size_t i = 0; i < B; ++i) d_b += grad_features_prev(i, v);
                grad_b_in_(0, v) += d_b;
            }
            for (size_t v = 0; v < V; ++v)
                for (size_t d = 0; d < (size_t)input_dim_; ++d) {
                    double s = 0.0;
                    for (size_t i = 0; i < B; ++i)
                        s += grad_features_prev(i, v) * last_input_(i, d);
                    grad_W_in_(v, d) += s;
                }
            for (size_t i = 0; i < B; ++i)
                for (size_t d = 0; d < (size_t)input_dim_; ++d) {
                    double s = 0.0;
                    for (size_t v = 0; v < V; ++v)
                        s += grad_features_prev(i, v) * W_in_(v, d);
                    grad_input(i, d) += s;
                }
        }
    }

    return grad_input;
}

// ============================================================================
// update_weights / zero_grad / accessors
// ============================================================================
void TabNet::update_weights(double learning_rate) {
    // Apply gradients
    for (size_t i = 0; i < param_list_.size(); ++i) {
        Tensor* p = param_list_[i];
        Tensor* g = grad_list_[i];
        for (size_t k = 0; k < p->data.size(); ++k)
            p->data[k] -= learning_rate * g->data[k];
    }
}

void TabNet::zero_grad() {
    for (auto* g : grad_list_) g->fill(0.0);
}

std::vector<Tensor*> TabNet::parameters() { return param_list_; }
std::vector<Tensor*> TabNet::gradients()  { return grad_list_; }

Tensor TabNet::getAttentionMask(int step) const {
    if (step < 0 || step >= (int)last_masks_.size()) return Tensor();
    return last_masks_[step];
}

std::vector<Tensor> TabNet::getAttentionMasks() const { return last_masks_; }
