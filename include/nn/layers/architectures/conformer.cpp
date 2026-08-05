#include "conformer.h"
#include "../../activations/activations.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// FeedForward
// ============================================================================
//
// Implements FFN(x) = Swish(W1 @ x + b1) @ W2 + b2
// Layout: input (dim, seq_len) -> output (dim, seq_len).
//
// Per-token (column-wise) math:
//   h[hi, t] = sum_d W1[hi, d] * x[d, t] + b1[0, hi]
//   h_act[hi, t] = h[hi, t] * sigmoid(h[hi, t])
//   out[d, t] = sum_hi W2[d, hi] * h_act[hi, t] + b2[0, d]
//
// Direct use of Dense is not feasible because Dense expects (batch, in_features)
// layout, while the rest of the Conformer block uses (d_model, seq_len) layout.
//
// Xavier-init for both W1 and W2: scale = 1/sqrt(in_dim).

FeedForward::FeedForward(size_t dim, size_t expansion)
    : dim_(dim),
      expansion_(expansion),
      W1_(dim * expansion, dim),
      b1_(1, dim * expansion),
      W2_(dim, dim * expansion),
      b2_(1, dim),
      grad_W1_(dim * expansion, dim),
      grad_b1_(1, dim * expansion),
      grad_W2_(dim, dim * expansion),
      grad_b2_(1, dim),
      last_input_(0, 0),
      last_h_pre_(0, 0),
      last_h_act_(0, 0)
{
    // Xavier init for both projections.
    double scale_w1 = 1.0 / std::sqrt(static_cast<double>(dim));
    double scale_w2 = 1.0 / std::sqrt(static_cast<double>(dim * expansion));
    std::mt19937 rng(42);
    std::normal_distribution<> d_w1(0.0, scale_w1);
    std::normal_distribution<> d_w2(0.0, scale_w2);
    for (size_t i = 0; i < W1_.data.size(); ++i) W1_.data[i] = d_w1(rng);
    for (size_t i = 0; i < W2_.data.size(); ++i) W2_.data[i] = d_w2(rng);
}

Tensor FeedForward::forward(const Tensor& input) {
    last_input_ = input.clone();
    size_t seq_len = input.cols;
    size_t hidden = dim_ * expansion_;

    Tensor h(hidden, seq_len);
    for (size_t hi = 0; hi < hidden; ++hi) {
        for (size_t t = 0; t < seq_len; ++t) {
            double v = b1_[0][hi];
            for (size_t d = 0; d < dim_; ++d) v += W1_[hi][d] * input[d][t];
            h[hi][t] = v;
        }
    }
    last_h_pre_ = h;

    Tensor h_act(hidden, seq_len);
    for (size_t i = 0; i < h.data.size(); ++i) {
        double hv = h.data[i];
        h_act.data[i] = hv / (1.0 + std::exp(-hv));
    }
    last_h_act_ = h_act;

    Tensor out(dim_, seq_len);
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t t = 0; t < seq_len; ++t) {
            double v = b2_[0][d];
            for (size_t hi = 0; hi < hidden; ++hi) v += W2_[d][hi] * h_act[hi][t];
            out[d][t] = v;
        }
    }
    return out;
}

Tensor FeedForward::backward(const Tensor& grad_output, double /* learning_rate */) {
    size_t seq_len = grad_output.cols;
    size_t hidden = dim_ * expansion_;

    // 1. d_h_act = W2^T @ d_out   (W2 is (dim, hidden), so W2^T is (hidden, dim))
    Tensor d_h_act(hidden, seq_len);
    for (size_t hi = 0; hi < hidden; ++hi) {
        for (size_t t = 0; t < seq_len; ++t) {
            double v = 0.0;
            for (size_t d = 0; d < dim_; ++d) v += W2_[d][hi] * grad_output[d][t];
            d_h_act[hi][t] = v;
        }
    }
    // 2. d_h_pre = d_h_act * swish'(h_pre)
    // swish'(z) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
    //           = sigmoid(z) * (1 + z - z*sigmoid(z))
    Tensor d_h_pre(hidden, seq_len);
    for (size_t i = 0; i < d_h_pre.data.size(); ++i) {
        double h_v = last_h_pre_.data[i];
        double sig_v = 1.0 / (1.0 + std::exp(-h_v));
        double silu_deriv = sig_v + h_v * sig_v * (1.0 - sig_v);
        d_h_pre.data[i] = d_h_act.data[i] * silu_deriv;
    }

    // 3. d_input = W1^T @ d_h_pre   (W1 is (hidden, dim), so W1^T is (dim, hidden))
    Tensor d_input(dim_, seq_len);
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t t = 0; t < seq_len; ++t) {
            double v = 0.0;
            for (size_t hi = 0; hi < hidden; ++hi) v += W1_[hi][d] * d_h_pre[hi][t];
            d_input[d][t] = v;
        }
    }

    // 4. Parameter gradients
    // grad_W1[hi, d] = sum_t d_h_pre[hi, t] * x[d, t]
    // dL/dW1[hi, d] = sum_t dL/d(h[hi, t]) * d(h[hi, t])/d(W1[hi, d]) = sum_t d_h_pre[hi, t] * x[d, t]
    for (size_t hi = 0; hi < hidden; ++hi) {
        for (size_t d = 0; d < dim_; ++d) {
            double s = 0.0;
            for (size_t t = 0; t < seq_len; ++t) s += d_h_pre[hi][t] * last_input_[d][t];
            grad_W1_[hi][d] += s;
        }
    }
    // grad_b1[0, hi] = sum_t d_h_pre[hi, t]
    for (size_t hi = 0; hi < hidden; ++hi) {
        double s = 0.0;
        for (size_t t = 0; t < seq_len; ++t) s += d_h_pre[hi][t];
        grad_b1_[0][hi] += s;
    }
    // grad_W2[d, hi] = sum_t d_out[d, t] * h_act[hi, t]
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t hi = 0; hi < hidden; ++hi) {
            double s = 0.0;
            for (size_t t = 0; t < seq_len; ++t) s += grad_output[d][t] * last_h_act_[hi][t];
            grad_W2_[d][hi] += s;
        }
    }
    // grad_b2[0, d] = sum_t d_out[d, t]
    for (size_t d = 0; d < dim_; ++d) {
        double s = 0.0;
        for (size_t t = 0; t < seq_len; ++t) s += grad_output[d][t];
        grad_b2_[0][d] += s;
    }

    return d_input;
}

void FeedForward::update_weights(double learning_rate) {
    for (size_t i = 0; i < W1_.data.size(); ++i) W1_.data[i] -= learning_rate * grad_W1_.data[i];
    for (size_t i = 0; i < b1_.data.size(); ++i) b1_.data[i] -= learning_rate * grad_b1_.data[i];
    for (size_t i = 0; i < W2_.data.size(); ++i) W2_.data[i] -= learning_rate * grad_W2_.data[i];
    for (size_t i = 0; i < b2_.data.size(); ++i) b2_.data[i] -= learning_rate * grad_b2_.data[i];
}

std::vector<Tensor*> FeedForward::parameters() { return {&W1_, &b1_, &W2_, &b2_}; }
std::vector<Tensor*> FeedForward::gradients() { return {&grad_W1_, &grad_b1_, &grad_W2_, &grad_b2_}; }

void FeedForward::zero_grad() {
    grad_W1_.fill(0.0);
    grad_b1_.fill(0.0);
    grad_W2_.fill(0.0);
    grad_b2_.fill(0.0);
}


// ============================================================================
// ConvModule
// ============================================================================
//
// Layout: input (dim, seq_len) -> output (dim, seq_len).
// Forward chain:
//   z1 = ln(x)                                  // (dim, seq_len)
//   z2 = pw_expand(z1)                          // (2*dim, seq_len)
//   [a, b] = split(z2, dim)                     // each (dim, seq_len)
//   z3 = a * sigmoid(b)                         // GLU, (dim, seq_len)
//   z4 = dw_conv(z3)                            // (dim, seq_len)
//   z5 = bn(z4)                                 // (dim, seq_len)
//   z6 = swish(z5)                              // (dim, seq_len)
//   z7 = pw_project(z6)                         // (dim, seq_len)
//
// Backward walks the chain in reverse.

ConvModule::ConvModule(size_t dim, size_t seq_len, size_t kernel_size)
    : ln_(dim),
      pw_expand_W_(2 * dim, dim),
      pw_expand_b_(1, 2 * dim),
      grad_pw_expand_W_(2 * dim, dim),
      grad_pw_expand_b_(1, 2 * dim),
      dw_conv_(static_cast<int>(dim), static_cast<int>(dim),
               static_cast<int>(kernel_size), static_cast<int>(seq_len),
               /*stride=*/1, /*pad=*/static_cast<int>(kernel_size / 2)),
      bn_(dim),
      pw_project_W_(dim, dim),
      pw_project_b_(1, dim),
      grad_pw_project_W_(dim, dim),
      grad_pw_project_b_(1, dim),
      dim_(dim),
      seq_len_(seq_len),
      kernel_size_(kernel_size),
      bn_training_(false)
{
    // Xavier init for pw_expand and pw_project (matmuls in (d_model, seq_len) layout).
    {
        double scale = 1.0 / std::sqrt(static_cast<double>(dim));
        std::mt19937 rng(101);
        std::normal_distribution<> d(0.0, scale);
        for (size_t i = 0; i < pw_expand_W_.data.size(); ++i) pw_expand_W_.data[i] = d(rng);
        for (size_t i = 0; i < pw_project_W_.data.size(); ++i) pw_project_W_.data[i] = d(rng);
    }

    // Initialize depthwise conv with identity-like center taps so that the conv
    // stage initially behaves as a passthrough.
    size_t mid = kernel_size / 2;
    for (size_t c = 0; c < dim_; ++c) {
        for (size_t j = 0; j < dw_conv_.weights.cols; ++j) {
            dw_conv_.weights[c][j] = 0.0;
        }
        dw_conv_.weights[c][c * kernel_size_ + mid] = 1.0;
        dw_conv_.bias[c][0] = 0.0;
    }

    // BN: training mode (uses input statistics for mean/var; matches what FD expects).
    // gamma is initialized to 1, beta to 0 — so BN normalizes but doesn't rescale.
    bn_.set_training(true);
    for (size_t i = 0; i < bn_.gamma.data.size(); ++i) bn_.gamma.data[i] = 1.0;
    for (size_t i = 0; i < bn_.beta.data.size(); ++i) bn_.beta.data[i] = 0.0;

    bn_training_ = true;
}

Tensor ConvModule::forward(const Tensor& input) {
    last_input_ = input.clone();

    // 1. LayerNorm
    Tensor z1 = ln_.forward(input);
    last_ln_out_ = z1;

    // 2. Pointwise expand: dim -> 2*dim
    // z2[hi, t] = sum_d pw_expand_W[hi, d] * z1[d, t] + pw_expand_b[0, hi]
    Tensor z2(2 * dim_, seq_len_);
    for (size_t hi = 0; hi < 2 * dim_; ++hi) {
        for (size_t t = 0; t < seq_len_; ++t) {
            double v = pw_expand_b_[0][hi];
            for (size_t d = 0; d < dim_; ++d) v += pw_expand_W_[hi][d] * z1[d][t];
            z2[hi][t] = v;
        }
    }
    last_expand_out_ = z2;

    // 3. GLU
    Tensor a(dim_, seq_len_), b(dim_, seq_len_);
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j < seq_len_; ++j) {
            a[i][j] = z2[i][j];
            b[i][j] = z2[dim_ + i][j];
        }
    }
    last_a_ = a;
    last_b_ = b;

    Tensor z3(dim_, seq_len_);
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j < seq_len_; ++j) {
            double sig = 1.0 / (1.0 + std::exp(-b[i][j]));
            z3[i][j] = a[i][j] * sig;
        }
    }
    last_glu_out_ = z3;

    // 4. Depthwise Conv1D.
    // Conv1D expects input shape (N=batch, in_channels * seq_len) flat, with the
    // data laid out as [c=0, t=0..seq_len-1, c=1, t=0..seq_len-1, ...] (channel-major
    // within each row). Our layout is (dim, seq_len) with [d, t] indexing.
    // Build the flat (1, dim*seq_len) tensor by iterating channels in the outer loop.
    Tensor z3_flat(1, dim_ * seq_len_);
    for (size_t c = 0; c < dim_; ++c) {
        for (size_t t = 0; t < seq_len_; ++t) {
            z3_flat[0][c * seq_len_ + t] = z3[c][t];
        }
    }
    Tensor z4_flat = dw_conv_.forward(z3_flat);  // (1, dim * seq_out)
    // z4_flat row 0 has channel-major layout. Unflatten back to (dim, seq_len).
    Tensor z4(dim_, seq_len_);
    for (size_t c = 0; c < dim_; ++c) {
        for (size_t t = 0; t < seq_len_; ++t) {
            z4[c][t] = z4_flat[0][c * seq_len_ + t];
        }
    }
    last_conv_out_ = z4;

    // 5. BatchNorm (BN expects (N, features) but our layout is (features, N).
    // Transpose, apply BN, transpose back).
    Tensor z4_t = z4.transpose();
    Tensor z5_t = bn_.forward(z4_t);
    Tensor z5 = z5_t.transpose();
    last_bn_out_ = z5;

    // 6. Swish (= SiLU)
    Tensor z6(z5.rows, z5.cols);
    for (size_t i = 0; i < z5.data.size(); ++i) {
        double v = z5.data[i];
        z6.data[i] = v / (1.0 + std::exp(-v));
    }
    last_silu_out_ = z6;

    // 7. Pointwise project: dim -> dim
    // z7[d, t] = sum_d' pw_project_W[d, d'] * z6[d', t] + pw_project_b[0, d]
    Tensor z7(dim_, seq_len_);
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t t = 0; t < seq_len_; ++t) {
            double v = pw_project_b_[0][d];
            for (size_t d2 = 0; d2 < dim_; ++d2) v += pw_project_W_[d][d2] * z6[d2][t];
            z7[d][t] = v;
        }
    }
    last_pw_project_in_ = z6;
    return z7;
}

Tensor ConvModule::backward(const Tensor& grad_output, double /* learning_rate */) {
    Tensor d_z7 = grad_output;

    // 1. pw_project backward
    // d_z6[d, t] = sum_d' pw_project_W[d', d] * d_z7[d', t]
    Tensor d_z6(dim_, seq_len_);
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t t = 0; t < seq_len_; ++t) {
            double v = 0.0;
            for (size_t d2 = 0; d2 < dim_; ++d2) v += pw_project_W_[d2][d] * d_z7[d2][t];
            d_z6[d][t] = v;
        }
    }
    // grad_pw_project_W[d, d2] += sum_t d_z7[d, t] * z6[d2, t]
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t d2 = 0; d2 < dim_; ++d2) {
            double s = 0.0;
            for (size_t t = 0; t < seq_len_; ++t) s += d_z7[d][t] * last_pw_project_in_[d2][t];
            grad_pw_project_W_[d][d2] += s;
        }
    }
    // grad_pw_project_b[0, d] += sum_t d_z7[d, t]
    for (size_t d = 0; d < dim_; ++d) {
        double s = 0.0;
        for (size_t t = 0; t < seq_len_; ++t) s += d_z7[d][t];
        grad_pw_project_b_[0][d] += s;
    }
    // 2. swish'(z5) = sigmoid(z5) + z5 * sigmoid(z5) * (1 - sigmoid(z5))
    // We use last_bn_out_ (z5) directly, not last_silu_out_ (z6 = swish(z5)).
    Tensor d_z5(d_z6.rows, d_z6.cols);
    for (size_t i = 0; i < d_z5.data.size(); ++i) {
        double z5_v = last_bn_out_.data[i];
        double sig_v = 1.0 / (1.0 + std::exp(-z5_v));
        double silu_deriv = sig_v + z5_v * sig_v * (1.0 - sig_v);
        d_z5.data[i] = d_z6.data[i] * silu_deriv;
    }
    Tensor d_z5_t = d_z5.transpose();
    Tensor d_z4_t = bn_.backward(d_z5_t, 0.0);
    Tensor d_z4 = d_z4_t.transpose();

    // 4. dw_conv backward.
    // Conv1D's backward expects grad_output in (N, out_channels * seq_out) flat form.
    // Our d_z4 is (dim, seq_len) — flatten it to (1, dim * seq_len) in channel-major.
    Tensor d_z4_flat(1, dim_ * seq_len_);
    for (size_t c = 0; c < dim_; ++c) {
        for (size_t t = 0; t < seq_len_; ++t) {
            d_z4_flat[0][c * seq_len_ + t] = d_z4[c][t];
        }
    }
    Tensor d_z3_flat = dw_conv_.backward(d_z4_flat, 0.0);  // (1, dim * seq_len)
    // Unflatten back to (dim, seq_len).
    Tensor d_z3(dim_, seq_len_);
    for (size_t c = 0; c < dim_; ++c) {
        for (size_t t = 0; t < seq_len_; ++t) {
            d_z3[c][t] = d_z3_flat[0][c * seq_len_ + t];
        }
    }

    // 5. GLU backward
    Tensor d_a(dim_, seq_len_), d_b(dim_, seq_len_);
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j < seq_len_; ++j) {
            double sig = 1.0 / (1.0 + std::exp(-last_b_[i][j]));
            d_a[i][j] = d_z3[i][j] * sig;
            d_b[i][j] = d_z3[i][j] * last_a_[i][j] * sig * (1.0 - sig);
        }
    }

    // 6. Stack d_a, d_b into d_z2 (2*dim, seq_len) and run pw_expand.backward
    Tensor d_z2(2 * dim_, seq_len_);
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j < seq_len_; ++j) {
            d_z2[i][j] = d_a[i][j];
            d_z2[dim_ + i][j] = d_b[i][j];
        }
    }

    // 7. pw_expand backward
    // d_z1[d, t] = sum_hi pw_expand_W[hi, d] * d_z2[hi, t]
    Tensor d_z1(dim_, seq_len_);
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t t = 0; t < seq_len_; ++t) {
            double v = 0.0;
            for (size_t hi = 0; hi < 2 * dim_; ++hi) v += pw_expand_W_[hi][d] * d_z2[hi][t];
            d_z1[d][t] = v;
        }
    }
    // grad_pw_expand_W[hi, d] += sum_t d_z2[hi, t] * z1[d, t]
    for (size_t hi = 0; hi < 2 * dim_; ++hi) {
        for (size_t d = 0; d < dim_; ++d) {
            double s = 0.0;
            for (size_t t = 0; t < seq_len_; ++t) s += d_z2[hi][t] * last_ln_out_[d][t];
            grad_pw_expand_W_[hi][d] += s;
        }
    }
    for (size_t hi = 0; hi < 2 * dim_; ++hi) {
        double s = 0.0;
        for (size_t t = 0; t < seq_len_; ++t) s += d_z2[hi][t];
        grad_pw_expand_b_[0][hi] += s;
    }

    // 8. LN backward
    Tensor d_input = ln_.backward(d_z1, 0.0);
    return d_input;
}

void ConvModule::update_weights(double learning_rate) {
    for (size_t i = 0; i < pw_expand_W_.data.size(); ++i)
        pw_expand_W_.data[i] -= learning_rate * grad_pw_expand_W_.data[i];
    for (size_t i = 0; i < pw_expand_b_.data.size(); ++i)
        pw_expand_b_.data[i] -= learning_rate * grad_pw_expand_b_.data[i];
    dw_conv_.update_weights(learning_rate);
    bn_.update_weights(learning_rate);
    for (size_t i = 0; i < pw_project_W_.data.size(); ++i)
        pw_project_W_.data[i] -= learning_rate * grad_pw_project_W_.data[i];
    for (size_t i = 0; i < pw_project_b_.data.size(); ++i)
        pw_project_b_.data[i] -= learning_rate * grad_pw_project_b_.data[i];
}

std::vector<Tensor*> ConvModule::parameters() {
    std::vector<Tensor*> result;
    auto p = ln_.parameters();
    result.insert(result.end(), p.begin(), p.end());
    result.push_back(&pw_expand_W_);
    result.push_back(&pw_expand_b_);
    auto p2 = dw_conv_.parameters();
    result.insert(result.end(), p2.begin(), p2.end());
    auto p3 = bn_.parameters();
    result.insert(result.end(), p3.begin(), p3.end());
    result.push_back(&pw_project_W_);
    result.push_back(&pw_project_b_);
    return result;
}

std::vector<Tensor*> ConvModule::gradients() {
    std::vector<Tensor*> result;
    auto g = ln_.gradients();
    result.insert(result.end(), g.begin(), g.end());
    result.push_back(&grad_pw_expand_W_);
    result.push_back(&grad_pw_expand_b_);
    auto g2 = dw_conv_.gradients();
    result.insert(result.end(), g2.begin(), g2.end());
    auto g3 = bn_.gradients();
    result.insert(result.end(), g3.begin(), g3.end());
    result.push_back(&grad_pw_project_W_);
    result.push_back(&grad_pw_project_b_);
    return result;
}

void ConvModule::zero_grad() {
    ln_.zero_grad();
    grad_pw_expand_W_.fill(0.0);
    grad_pw_expand_b_.fill(0.0);
    dw_conv_.zero_grad();
    bn_.zero_grad();
    grad_pw_project_W_.fill(0.0);
    grad_pw_project_b_.fill(0.0);
}


// ============================================================================
// ConformerBlock
// ============================================================================

ConformerBlock::ConformerBlock(size_t dim, size_t num_heads, size_t seq_len,
                               size_t ffn_expansion, size_t conv_kernel_size)
    : dim_(dim),
      seq_len_(seq_len),
      ln_1_(dim),
      ffn1_(dim, ffn_expansion),
      ln_2_(dim),
      mhsa_(dim, num_heads),
      ln_3_(dim),
      conv_(dim, seq_len, conv_kernel_size),
      ln_4_(dim),
      ffn2_(dim, ffn_expansion),
      ln_5_(dim)
{
}

Tensor ConformerBlock::forward(const Tensor& input) {
    last_input_ = input.clone();

    Tensor z1 = ln_1_.forward(input);
    last_ln1_out_ = z1;
    Tensor z2 = ffn1_.forward(z1);
    last_ffn1_out_ = z2;
    Tensor r1(input.rows, input.cols);
    for (size_t i = 0; i < r1.data.size(); ++i) r1.data[i] = input.data[i] + 0.5 * z2.data[i];
    last_r1_ = r1;

    Tensor z3 = ln_2_.forward(r1);
    last_ln2_out_ = z3;
    Tensor z4 = mhsa_.forward(z3);
    last_mhsa_out_ = z4;
    Tensor r2(r1.rows, r1.cols);
    for (size_t i = 0; i < r2.data.size(); ++i) r2.data[i] = r1.data[i] + z4.data[i];
    last_r2_ = r2;

    Tensor z5 = ln_3_.forward(r2);
    last_ln3_out_ = z5;
    Tensor z6 = conv_.forward(z5);
    last_conv_out_ = z6;
    Tensor r3(r2.rows, r2.cols);
    for (size_t i = 0; i < r3.data.size(); ++i) r3.data[i] = r2.data[i] + z6.data[i];
    last_r3_ = r3;

    Tensor z7 = ln_4_.forward(r3);
    last_ln4_out_ = z7;
    Tensor z8 = ffn2_.forward(z7);
    last_ffn2_out_ = z8;
    Tensor r4(r3.rows, r3.cols);
    for (size_t i = 0; i < r4.data.size(); ++i) r4.data[i] = r3.data[i] + 0.5 * z8.data[i];
    last_r4_ = r4;

    Tensor out = ln_5_.forward(r4);
    last_output_ = out;
    return out;
}

Tensor ConformerBlock::backward(const Tensor& grad_output, double /* learning_rate */) {
    Tensor d_r4 = ln_5_.backward(grad_output, 0.0);

    Tensor d_r3 = d_r4;
    Tensor d_z8(d_r4.rows, d_r4.cols);
    for (size_t i = 0; i < d_z8.data.size(); ++i) d_z8.data[i] = 0.5 * d_r4.data[i];

    Tensor d_z7 = ffn2_.backward(d_z8, 0.0);
    Tensor d_ln4_out = d_z7;
    Tensor d_r3_from_ln4 = ln_4_.backward(d_ln4_out, 0.0);
    for (size_t i = 0; i < d_r3.data.size(); ++i) d_r3.data[i] += d_r3_from_ln4.data[i];

    Tensor d_r2 = d_r3;
    Tensor d_z6 = d_r3;

    Tensor d_z5 = conv_.backward(d_z6, 0.0);
    Tensor d_ln3_out = d_z5;
    Tensor d_r2_from_ln3 = ln_3_.backward(d_ln3_out, 0.0);
    for (size_t i = 0; i < d_r2.data.size(); ++i) d_r2.data[i] += d_r2_from_ln3.data[i];

    Tensor d_r1 = d_r2;
    Tensor d_z4 = d_r2;

    Tensor d_z3 = mhsa_.backward(d_z4, 0.0);
    Tensor d_ln2_out = d_z3;
    Tensor d_r1_from_ln2 = ln_2_.backward(d_ln2_out, 0.0);
    for (size_t i = 0; i < d_r1.data.size(); ++i) d_r1.data[i] += d_r1_from_ln2.data[i];

    Tensor d_x = d_r1;
    Tensor d_z2(d_r1.rows, d_r1.cols);
    for (size_t i = 0; i < d_z2.data.size(); ++i) d_z2.data[i] = 0.5 * d_r1.data[i];

    Tensor d_z1 = ffn1_.backward(d_z2, 0.0);
    Tensor d_ln1_out = d_z1;
    Tensor d_x_from_ln1 = ln_1_.backward(d_ln1_out, 0.0);
    for (size_t i = 0; i < d_x.data.size(); ++i) d_x.data[i] += d_x_from_ln1.data[i];

    return d_x;
}

void ConformerBlock::update_weights(double learning_rate) {
    ln_1_.update_weights(learning_rate);
    ffn1_.update_weights(learning_rate);
    ln_2_.update_weights(learning_rate);
    mhsa_.update_weights(learning_rate);
    ln_3_.update_weights(learning_rate);
    conv_.update_weights(learning_rate);
    ln_4_.update_weights(learning_rate);
    ffn2_.update_weights(learning_rate);
    ln_5_.update_weights(learning_rate);
}

std::vector<Tensor*> ConformerBlock::parameters() {
    std::vector<Tensor*> result;
    auto p = ln_1_.parameters(); result.insert(result.end(), p.begin(), p.end());
    p = ffn1_.parameters(); result.insert(result.end(), p.begin(), p.end());
    p = ln_2_.parameters(); result.insert(result.end(), p.begin(), p.end());
    p = mhsa_.parameters(); result.insert(result.end(), p.begin(), p.end());
    p = ln_3_.parameters(); result.insert(result.end(), p.begin(), p.end());
    p = conv_.parameters(); result.insert(result.end(), p.begin(), p.end());
    p = ln_4_.parameters(); result.insert(result.end(), p.begin(), p.end());
    p = ffn2_.parameters(); result.insert(result.end(), p.begin(), p.end());
    p = ln_5_.parameters(); result.insert(result.end(), p.begin(), p.end());
    return result;
}

std::vector<Tensor*> ConformerBlock::gradients() {
    std::vector<Tensor*> result;
    auto g = ln_1_.gradients(); result.insert(result.end(), g.begin(), g.end());
    g = ffn1_.gradients(); result.insert(result.end(), g.begin(), g.end());
    g = ln_2_.gradients(); result.insert(result.end(), g.begin(), g.end());
    g = mhsa_.gradients(); result.insert(result.end(), g.begin(), g.end());
    g = ln_3_.gradients(); result.insert(result.end(), g.begin(), g.end());
    g = conv_.gradients(); result.insert(result.end(), g.begin(), g.end());
    g = ln_4_.gradients(); result.insert(result.end(), g.begin(), g.end());
    g = ffn2_.gradients(); result.insert(result.end(), g.begin(), g.end());
    g = ln_5_.gradients(); result.insert(result.end(), g.begin(), g.end());
    return result;
}

void ConformerBlock::zero_grad() {
    ln_1_.zero_grad();
    ffn1_.zero_grad();
    ln_2_.zero_grad();
    mhsa_.zero_grad();
    ln_3_.zero_grad();
    conv_.zero_grad();
    ln_4_.zero_grad();
    ffn2_.zero_grad();
    ln_5_.zero_grad();
}


// ============================================================================
// ConformerModel
// ============================================================================

ConformerModel::ConformerModel(size_t input_dim, size_t num_classes,
                               size_t dim, size_t depth, size_t num_heads,
                               size_t seq_len, size_t ffn_expansion,
                               size_t conv_kernel_size)
    : input_dim_(input_dim),
      num_classes_(num_classes),
      dim_(dim),
      depth_(depth),
      num_heads_(num_heads),
      seq_len_(seq_len),
      ffn_expansion_(ffn_expansion),
      conv_kernel_size_(conv_kernel_size),
      input_proj_W_(dim, input_dim),
      input_proj_b_(1, dim),
      grad_input_proj_W_(dim, input_dim),
      grad_input_proj_b_(1, dim),
      blocks_(),
      ln_out_(dim),
      classifier_W_(num_classes, dim),
      classifier_b_(1, num_classes),
      grad_classifier_W_(num_classes, dim),
      grad_classifier_b_(1, num_classes)
{
    blocks_.reserve(depth_);
    for (size_t i = 0; i < depth_; ++i) {
        blocks_.emplace_back(dim, num_heads, seq_len, ffn_expansion, conv_kernel_size);
    }
    last_block_outs_.reserve(depth_);

    // Xavier init for input projection and classifier.
    {
        std::mt19937 rng(303);
        std::normal_distribution<> d_in(0.0, 1.0 / std::sqrt(static_cast<double>(input_dim)));
        std::normal_distribution<> d_out(0.0, 1.0 / std::sqrt(static_cast<double>(dim)));
        for (size_t i = 0; i < input_proj_W_.data.size(); ++i) input_proj_W_.data[i] = d_in(rng);
        for (size_t i = 0; i < classifier_W_.data.size(); ++i) classifier_W_.data[i] = d_out(rng);
    }
}

Tensor ConformerModel::forward(const Tensor& input) {
    last_input_ = input.clone();

    // input projection: (input_dim, seq_len) -> (dim, seq_len)
    Tensor proj(dim_, input.cols);
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t t = 0; t < input.cols; ++t) {
            double v = input_proj_b_[0][d];
            for (size_t d2 = 0; d2 < input_dim_; ++d2) v += input_proj_W_[d][d2] * input[d2][t];
            proj[d][t] = v;
        }
    }
    last_proj_out_ = proj;

    // Stack of ConformerBlocks
    Tensor h = proj;
    last_block_outs_.clear();
    for (size_t i = 0; i < depth_; ++i) {
        h = blocks_[i].forward(h);
        last_block_outs_.push_back(h);
    }

    // Mean pool over time (columns)
    Tensor pooled(dim_, 1);
    for (size_t i = 0; i < dim_; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < seq_len_; ++j) sum += h[i][j];
        pooled[i][0] = sum / static_cast<double>(seq_len_);
    }
    last_pool_ = pooled;

    Tensor ln_out = ln_out_.forward(pooled);
    last_ln_out_ = ln_out;

    // Classifier: (dim, 1) -> (num_classes, 1)
    Tensor out(num_classes_, 1);
    for (size_t c = 0; c < num_classes_; ++c) {
        double v = classifier_b_[0][c];
        for (size_t d = 0; d < dim_; ++d) v += classifier_W_[c][d] * ln_out[d][0];
        out[c][0] = v;
    }
    last_output_ = out;
    return out;
}

Tensor ConformerModel::backward(const Tensor& grad_output, double /* learning_rate */) {
    // grad_output: (num_classes_, 1)

    // 1. Classifier backward
    // d_ln_out[d, 0] = sum_c classifier_W[c, d] * grad_output[c, 0]
    Tensor d_ln_out(dim_, 1);
    for (size_t d = 0; d < dim_; ++d) {
        double v = 0.0;
        for (size_t c = 0; c < num_classes_; ++c) v += classifier_W_[c][d] * grad_output[c][0];
        d_ln_out[d][0] = v;
    }
    // grad_classifier_W[c, d] += grad_output[c, 0] * ln_out[d, 0]
    for (size_t c = 0; c < num_classes_; ++c) {
        for (size_t d = 0; d < dim_; ++d) {
            grad_classifier_W_[c][d] += grad_output[c][0] * last_ln_out_[d][0];
        }
    }
    for (size_t c = 0; c < num_classes_; ++c) {
        grad_classifier_b_[0][c] += grad_output[c][0];
    }

    // 2. LN_out backward
    Tensor d_pool = ln_out_.backward(d_ln_out, 0.0);

    // 3. Broadcast d_pool across time (mean pool was (1/seq_len) over each col)
    Tensor d_block_out_last(dim_, seq_len_);
    for (size_t i = 0; i < dim_; ++i) {
        for (size_t j = 0; j < seq_len_; ++j) {
            d_block_out_last[i][j] = d_pool[i][0] / static_cast<double>(seq_len_);
        }
    }

    // 4. Backward through blocks in reverse order
    Tensor d_block_in = d_block_out_last;
    for (size_t i = depth_; i > 0; --i) {
        d_block_in = blocks_[i - 1].backward(d_block_in, 0.0);
    }

    // 5. input_proj backward
    Tensor d_input(input_dim_, seq_len_);
    for (size_t d = 0; d < input_dim_; ++d) {
        for (size_t t = 0; t < seq_len_; ++t) {
            double v = 0.0;
            for (size_t d2 = 0; d2 < dim_; ++d2) v += input_proj_W_[d2][d] * d_block_in[d2][t];
            d_input[d][t] = v;
        }
    }
    // grad_input_proj_W[d, d2] += sum_t d_block_in[d, t] * input[d2, t]
    for (size_t d = 0; d < dim_; ++d) {
        for (size_t d2 = 0; d2 < input_dim_; ++d2) {
            double s = 0.0;
            for (size_t t = 0; t < seq_len_; ++t) s += d_block_in[d][t] * last_input_[d2][t];
            grad_input_proj_W_[d][d2] += s;
        }
    }
    for (size_t d = 0; d < dim_; ++d) {
        double s = 0.0;
        for (size_t t = 0; t < seq_len_; ++t) s += d_block_in[d][t];
        grad_input_proj_b_[0][d] += s;
    }

    return d_input;
}

void ConformerModel::update_weights(double learning_rate) {
    for (size_t i = 0; i < input_proj_W_.data.size(); ++i)
        input_proj_W_.data[i] -= learning_rate * grad_input_proj_W_.data[i];
    for (size_t i = 0; i < input_proj_b_.data.size(); ++i)
        input_proj_b_.data[i] -= learning_rate * grad_input_proj_b_.data[i];
    for (size_t i = 0; i < depth_; ++i) blocks_[i].update_weights(learning_rate);
    ln_out_.update_weights(learning_rate);
    for (size_t i = 0; i < classifier_W_.data.size(); ++i)
        classifier_W_.data[i] -= learning_rate * grad_classifier_W_.data[i];
    for (size_t i = 0; i < classifier_b_.data.size(); ++i)
        classifier_b_.data[i] -= learning_rate * grad_classifier_b_.data[i];
}

std::vector<Tensor*> ConformerModel::parameters() {
    std::vector<Tensor*> result;
    result.push_back(&input_proj_W_);
    result.push_back(&input_proj_b_);
    for (size_t i = 0; i < depth_; ++i) {
        auto p = blocks_[i].parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    auto p = ln_out_.parameters();
    result.insert(result.end(), p.begin(), p.end());
    result.push_back(&classifier_W_);
    result.push_back(&classifier_b_);
    return result;
}

std::vector<Tensor*> ConformerModel::gradients() {
    std::vector<Tensor*> result;
    result.push_back(&grad_input_proj_W_);
    result.push_back(&grad_input_proj_b_);
    for (size_t i = 0; i < depth_; ++i) {
        auto g = blocks_[i].gradients();
        result.insert(result.end(), g.begin(), g.end());
    }
    auto g = ln_out_.gradients();
    result.insert(result.end(), g.begin(), g.end());
    result.push_back(&grad_classifier_W_);
    result.push_back(&grad_classifier_b_);
    return result;
}

void ConformerModel::zero_grad() {
    grad_input_proj_W_.fill(0.0);
    grad_input_proj_b_.fill(0.0);
    for (size_t i = 0; i < depth_; ++i) blocks_[i].zero_grad();
    ln_out_.zero_grad();
    grad_classifier_W_.fill(0.0);
    grad_classifier_b_.fill(0.0);
}
