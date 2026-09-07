// Stick-Breaking Attention — Tan et al., ICLR 2025
//   "Scaling Stick-Breaking Attention: An Efficient Implementation and
//    In-depth Study"
//   https://arxiv.org/abs/2410.17980
//
// See stick_breaking.h for the full formulation and backward derivation.
//
// Implementation notes (the bits easy to get wrong):
//   * Log-space forward (Eq. 13): A[i,j] = exp(z[i,j] - Σ_{k=i}^{j-1} sp(z[k,j])).
//     Walking i downward from j-1 lets the suffix sum Σ_{k=i}^{j-1} be
//     accumulated incrementally — O(L²) total, no underflow.
//   * Backward uses the PREFIX sum identity
//       dz[m,j] = dS[m,j] - σ(z[m,j]) * Σ_{i=0}^{m} dS[i,j]
//     accumulated as m increases (again O(L²) rather than the naive O(L³)).
//   * Strict causality: i < j. Row j=0 has an all-zero attention row, so with
//     the remainder bias o_0 == r_head exactly.
//   * Projection parameter grads go into raw grad_W_* tensors (not via
//     Dense::backward) because the per-head chain needs manual index control.

#include "stick_breaking.h"
#include <cmath>
#include <random>
#include <stdexcept>

namespace {

// Numerically stable softplus: log(1 + exp(x)).
inline double sb_softplus(double x) {
    return std::max(x, 0.0) + std::log1p(std::exp(-std::fabs(x)));
}

inline double sb_sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    double e = std::exp(x);
    return e / (1.0 + e);
}

inline double sb_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double sb_gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

} // namespace

// ============================================================================
// StickBreakingAttention
// ============================================================================

StickBreakingAttention::StickBreakingAttention(size_t d_model, size_t num_heads,
                                               bool use_remainder)
    : W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      W_o(d_model ? d_model : 1, d_model ? d_model : 1),
      remainder_(num_heads ? num_heads : 1,
                 (num_heads && d_model && d_model % num_heads == 0)
                     ? d_model / num_heads : 1),
      grad_W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_o(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_remainder_(num_heads ? num_heads : 1,
                      (num_heads && d_model && d_model % num_heads == 0)
                          ? d_model / num_heads : 1),
      d_model_(d_model), num_heads_(num_heads),
      head_dim_((num_heads && d_model && d_model % num_heads == 0)
                    ? d_model / num_heads : 1),
      use_remainder_(use_remainder)
{
    if (d_model == 0)
        throw std::invalid_argument("StickBreakingAttention: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("StickBreakingAttention: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument(
            "StickBreakingAttention: d_model must be divisible by num_heads");

    inv_temp_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_remainder_.fill(0.0);

    // The remainder embedding starts at zero: initially the leftover stick
    // contributes nothing, so the layer reduces to the bare Eq. 1 form and
    // learns the sink from there.
    remainder_.fill(0.0);
}

Tensor StickBreakingAttention::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument(
            "StickBreakingAttention: input.cols must equal d_model");
    if (input.rows == 0)
        throw std::invalid_argument("StickBreakingAttention: input.rows must be > 0");

    const size_t N = input.rows;
    const size_t H = num_heads_;
    const size_t dh = head_dim_;
    N_last_ = N;
    last_input_ = input.clone();

    // Projections: Q = X @ W_q^T + b, etc. (Dense stores (out, in)).
    last_Q_ = W_q.forward(input);
    last_K_ = W_k.forward(input);
    last_V_ = W_v.forward(input);

    last_Z_   = Tensor(H * N, N);
    last_A_   = Tensor(H * N, N);
    last_rem_ = Tensor(H, N);
    last_Z_.fill(0.0);
    last_A_.fill(0.0);
    last_O_ = Tensor(N, d_model_);
    last_O_.fill(0.0);

    for (size_t h = 0; h < H; ++h) {
        const size_t off = h * dh;
        for (size_t j = 0; j < N; ++j) {
            const size_t zrow = h * N + j;

            // Logits z[i,j] = inv_temp * (q_j · k_i) for i < j.
            for (size_t i = 0; i < j; ++i) {
                double dot = 0.0;
                for (size_t d = 0; d < dh; ++d)
                    dot += last_Q_(j, off + d) * last_K_(i, off + d);
                last_Z_(zrow, i) = dot * inv_temp_;
            }

            // Log-space stick-breaking (Eq. 13). Walk i downward from j-1 so
            // the suffix sum Σ_{k=i}^{j-1} softplus(z[k,j]) accumulates.
            double suffix_sp = 0.0;   // Σ_{k=i+1}^{j-1} softplus(z[k,j])
            double a_sum = 0.0;
            for (size_t ip1 = j; ip1 > 0; --ip1) {
                const size_t i = ip1 - 1;
                const double z = last_Z_(zrow, i);
                // A[i,j] = exp(z - softplus(z) - Σ_{k>i} softplus(z[k,j]))
                const double a = std::exp(z - sb_softplus(z) - suffix_sp);
                last_A_(zrow, i) = a;
                a_sum += a;
                suffix_sp += sb_softplus(z);
            }

            double rem = 1.0 - a_sum;
            // Guard against tiny negative values from FP cancellation.
            if (rem < 0.0 && rem > -1e-12) rem = 0.0;
            last_rem_(h, j) = rem;

            // o_j = Σ_{i<j} A[i,j] v_i  (+ rem_j · r_head)
            for (size_t d = 0; d < dh; ++d) {
                double acc = 0.0;
                for (size_t i = 0; i < j; ++i)
                    acc += last_A_(zrow, i) * last_V_(i, off + d);
                if (use_remainder_) acc += rem * remainder_(h, d);
                last_O_(j, off + d) = acc;
            }
        }
    }

    return W_o.forward(last_O_);
}

Tensor StickBreakingAttention::backward(const Tensor& grad_output,
                                        double /* learning_rate */) {
    const size_t N = N_last_;
    const size_t H = num_heads_;
    const size_t dh = head_dim_;

    if (grad_output.rows != N || grad_output.cols != d_model_)
        throw std::invalid_argument(
            "StickBreakingAttention::backward: grad_output shape mismatch");

    // --- Output projection: Y = O @ W_o^T + b_o ---
    // dO = dY @ W_o ; dW_o += dY^T @ O ; db_o += colsum(dY)
    Tensor dO(N, d_model_);
    dO.fill(0.0);
    for (size_t t = 0; t < N; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t i = 0; i < d_model_; ++i)
                acc += grad_output(t, i) * W_o.weights(i, j);
            dO(t, j) = acc;
        }
    for (size_t i = 0; i < d_model_; ++i) {
        double bacc = 0.0;
        for (size_t t = 0; t < N; ++t) {
            bacc += grad_output(t, i);
            for (size_t j = 0; j < d_model_; ++j)
                grad_W_o(i, j) += grad_output(t, i) * last_O_(t, j);
        }
        W_o.grad_bias(0, i) += bacc;
    }

    // --- Per-head stick-breaking backward ---
    Tensor dQ(N, d_model_), dK(N, d_model_), dV(N, d_model_);
    dQ.fill(0.0); dK.fill(0.0); dV.fill(0.0);

    for (size_t h = 0; h < H; ++h) {
        const size_t off = h * dh;
        for (size_t j = 0; j < N; ++j) {
            const size_t zrow = h * N + j;

            // dr += rem_j * dO_j
            if (use_remainder_) {
                const double rem = last_rem_(h, j);
                for (size_t d = 0; d < dh; ++d)
                    grad_remainder_(h, d) += rem * dO(j, off + d);
            }

            // dS[i,j] = A[i,j] * (dO_j·v_i - dO_j·r_head)
            //   the 2nd term comes from rem_j = 1 - Σ_i A[i,j]
            // dV[i]  += A[i,j] * dO_j
            std::vector<double> dS(j, 0.0);
            double dot_r = 0.0;
            if (use_remainder_)
                for (size_t d = 0; d < dh; ++d)
                    dot_r += dO(j, off + d) * remainder_(h, d);

            for (size_t i = 0; i < j; ++i) {
                const double a = last_A_(zrow, i);
                double dot_v = 0.0;
                for (size_t d = 0; d < dh; ++d) {
                    const double g = dO(j, off + d);
                    dot_v += g * last_V_(i, off + d);
                    dV(i, off + d) += a * g;
                }
                dS[i] = a * (dot_v - dot_r);
            }

            // dz[m,j] = dS[m,j] - σ(z[m,j]) * Σ_{i=0}^{m} dS[i,j]
            // (prefix sum over i, accumulated as m increases)
            double prefix = 0.0;
            for (size_t m = 0; m < j; ++m) {
                prefix += dS[m];
                const double dz = dS[m] - sb_sigmoid(last_Z_(zrow, m)) * prefix;
                // z[m,j] = inv_temp * (q_j · k_m)
                const double s = dz * inv_temp_;
                for (size_t d = 0; d < dh; ++d) {
                    dQ(j, off + d) += s * last_K_(m, off + d);
                    dK(m, off + d) += s * last_Q_(j, off + d);
                }
            }
        }
    }

    // --- Projection params + input gradient ---
    // For Dense W (out, in): Y = X @ W^T + b
    //   dW += dY^T @ X ; db += colsum(dY) ; dX += dY @ W
    Tensor d_input(N, d_model_);
    d_input.fill(0.0);

    auto accum_proj = [&](const Tensor& dY, const Dense& W, Tensor& gradW,
                          Tensor& gradB) {
        for (size_t i = 0; i < d_model_; ++i) {
            double bacc = 0.0;
            for (size_t t = 0; t < N; ++t) {
                const double g = dY(t, i);
                bacc += g;
                for (size_t j = 0; j < d_model_; ++j)
                    gradW(i, j) += g * last_input_(t, j);
            }
            gradB(0, i) += bacc;
        }
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d_model_; ++j) {
                double acc = 0.0;
                for (size_t i = 0; i < d_model_; ++i)
                    acc += dY(t, i) * W.weights(i, j);
                d_input(t, j) += acc;
            }
    };

    accum_proj(dQ, W_q, grad_W_q, W_q.grad_bias);
    accum_proj(dK, W_k, grad_W_k, W_k.grad_bias);
    accum_proj(dV, W_v, grad_W_v, W_v.grad_bias);

    return d_input;
}

void StickBreakingAttention::update_weights(double lr) {
    W_q.weights -= grad_W_q * lr;
    W_k.weights -= grad_W_k * lr;
    W_v.weights -= grad_W_v * lr;
    W_o.weights -= grad_W_o * lr;
    W_q.bias -= W_q.grad_bias * lr;
    W_k.bias -= W_k.grad_bias * lr;
    W_v.bias -= W_v.grad_bias * lr;
    W_o.bias -= W_o.grad_bias * lr;
    if (use_remainder_) remainder_ -= grad_remainder_ * lr;
}

void StickBreakingAttention::zero_grad() {
    grad_W_q.fill(0.0); W_q.grad_bias.fill(0.0);
    grad_W_k.fill(0.0); W_k.grad_bias.fill(0.0);
    grad_W_v.fill(0.0); W_v.grad_bias.fill(0.0);
    grad_W_o.fill(0.0); W_o.grad_bias.fill(0.0);
    grad_remainder_.fill(0.0);
}

std::vector<Tensor*> StickBreakingAttention::parameters() {
    return {&W_q.weights, &W_q.bias,
            &W_k.weights, &W_k.bias,
            &W_v.weights, &W_v.bias,
            &W_o.weights, &W_o.bias,
            &remainder_};
}

std::vector<Tensor*> StickBreakingAttention::gradients() {
    return {&grad_W_q, &W_q.grad_bias,
            &grad_W_k, &W_k.grad_bias,
            &grad_W_v, &W_v.grad_bias,
            &grad_W_o, &W_o.grad_bias,
            &grad_remainder_};
}

// ============================================================================
// StickBreakingBlock
// ============================================================================

StickBreakingBlock::StickBreakingBlock(size_t d_model, size_t num_heads,
                                       size_t ffn_dim, bool use_remainder)
    : attn(d_model, num_heads, use_remainder),
      ln1(d_model ? d_model : 1),
      ln2(d_model ? d_model : 1),
      ffn_fc1_(d_model ? d_model : 1,
               ffn_dim ? ffn_dim : (d_model ? 4 * d_model : 1)),
      ffn_fc2_(ffn_dim ? ffn_dim : (d_model ? 4 * d_model : 1),
               d_model ? d_model : 1),
      d_model_(d_model),
      ffn_dim_(ffn_dim ? ffn_dim : 4 * d_model)
{
    if (d_model == 0)
        throw std::invalid_argument("StickBreakingBlock: d_model must be > 0");
}

Tensor StickBreakingBlock::forward(const Tensor& input) {
    last_x_ = input.clone();
    last_z1_ = ln1.forward(input);
    last_attn_out_ = attn.forward(last_z1_);
    last_res1_ = last_z1_ + last_attn_out_;
    last_z2_ = ln2.forward(last_res1_);
    last_h_pre_ = ffn_fc1_.forward(last_z2_);
    last_h_act_ = last_h_pre_.apply(sb_gelu);
    Tensor ffn_out = ffn_fc2_.forward(last_h_act_);
    return last_res1_ + ffn_out;
}

Tensor StickBreakingBlock::backward(const Tensor& grad_output, double lr) {
    // out = res1 + ffn_out  ->  d_res1 = grad_output + (chain through FFN)
    Tensor d_h_act = ffn_fc2_.backward(grad_output, lr);
    Tensor d_h_pre(d_h_act.rows, d_h_act.cols);
    for (size_t i = 0; i < d_h_act.rows; ++i)
        for (size_t j = 0; j < d_h_act.cols; ++j)
            d_h_pre(i, j) = d_h_act(i, j) * sb_gelu_deriv(last_h_pre_(i, j));
    Tensor d_z2 = ffn_fc1_.backward(d_h_pre, lr);

    // z2 = ln2(res1) -> must route d_z2 THROUGH ln2, not add it directly.
    Tensor d_res1 = grad_output + ln2.backward(d_z2, lr);

    // res1 = z1 + attn_out
    Tensor d_z1 = d_res1 + attn.backward(d_res1, lr);

    // z1 = ln1(x)
    return ln1.backward(d_z1, lr);
}

void StickBreakingBlock::update_weights(double lr) {
    attn.update_weights(lr);
    ln1.update_weights(lr);
    ln2.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

void StickBreakingBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> StickBreakingBlock::parameters() {
    std::vector<Tensor*> p = attn.parameters();
    for (Tensor* t : ln1.parameters())      p.push_back(t);
    for (Tensor* t : ln2.parameters())      p.push_back(t);
    for (Tensor* t : ffn_fc1_.parameters()) p.push_back(t);
    for (Tensor* t : ffn_fc2_.parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> StickBreakingBlock::gradients() {
    std::vector<Tensor*> g = attn.gradients();
    for (Tensor* t : ln1.gradients())      g.push_back(t);
    for (Tensor* t : ln2.gradients())      g.push_back(t);
    for (Tensor* t : ffn_fc1_.gradients()) g.push_back(t);
    for (Tensor* t : ffn_fc2_.gradients()) g.push_back(t);
    return g;
}

// ============================================================================
// StickBreakingModel
// ============================================================================

StickBreakingModel::StickBreakingModel(size_t input_dim, size_t d_model,
                                       size_t output_dim, size_t num_blocks,
                                       size_t num_heads, bool use_remainder)
    : input_proj(input_dim ? input_dim : 1, d_model ? d_model : 1),
      final_ln(d_model ? d_model : 1),
      classifier(d_model ? d_model : 1, output_dim ? output_dim : 1),
      input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim),
      num_blocks_(num_blocks)
{
    if (input_dim == 0)
        throw std::invalid_argument("StickBreakingModel: input_dim must be > 0");
    if (d_model == 0)
        throw std::invalid_argument("StickBreakingModel: d_model must be > 0");
    if (output_dim == 0)
        throw std::invalid_argument("StickBreakingModel: output_dim must be > 0");
    if (num_blocks == 0)
        throw std::invalid_argument("StickBreakingModel: num_blocks must be > 0");

    for (size_t b = 0; b < num_blocks; ++b)
        blocks.push_back(std::make_unique<StickBreakingBlock>(
            d_model, num_heads, 0, use_remainder));
}

Tensor StickBreakingModel::forward(const Tensor& input) {
    Tensor h = input_proj.forward(input);
    for (auto& blk : blocks) h = blk->forward(h);
    h = final_ln.forward(h);
    return classifier.forward(h);
}

Tensor StickBreakingModel::backward(const Tensor& grad_output, double lr) {
    Tensor g = classifier.backward(grad_output, lr);
    g = final_ln.backward(g, lr);
    for (size_t i = blocks.size(); i > 0; --i)
        g = blocks[i - 1]->backward(g, lr);
    return input_proj.backward(g, lr);
}

void StickBreakingModel::update_weights(double lr) {
    input_proj.update_weights(lr);
    for (auto& blk : blocks) blk->update_weights(lr);
    final_ln.update_weights(lr);
    classifier.update_weights(lr);
}

void StickBreakingModel::zero_grad() {
    input_proj.zero_grad();
    for (auto& blk : blocks) blk->zero_grad();
    final_ln.zero_grad();
    classifier.zero_grad();
}

std::vector<Tensor*> StickBreakingModel::parameters() {
    std::vector<Tensor*> p = input_proj.parameters();
    for (auto& blk : blocks)
        for (Tensor* t : blk->parameters()) p.push_back(t);
    for (Tensor* t : final_ln.parameters())   p.push_back(t);
    for (Tensor* t : classifier.parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> StickBreakingModel::gradients() {
    std::vector<Tensor*> g = input_proj.gradients();
    for (auto& blk : blocks)
        for (Tensor* t : blk->gradients()) g.push_back(t);
    for (Tensor* t : final_ln.gradients())   g.push_back(t);
    for (Tensor* t : classifier.gradients()) g.push_back(t);
    return g;
}
