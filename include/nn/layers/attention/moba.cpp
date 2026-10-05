// MoBA — Mixture of Block Attention (Lu et al., arXiv:2502.13189)
//
// See moba.h for the full formulation, the paper-equation mapping, and the
// backward derivation.
//
// Implementation notes (the bits easy to get wrong):
//   * The gate score is UNSCALED. Eq. 6 / Algorithm 1 line 5 have no 1/sqrt(d_h)
//     factor — inv_temp_ applies to the attention logits only. Scaling the gate
//     too makes every score 2x wrong at head_dim=4.
//   * mean_pool, not sum, per Algorithm 1 line 4. Sum and mean only rescale
//     the top-k ORDER (so selection is unaffected) but differ by block_size in
//     the score and in the K gradient.
//   * top_k_ counts HISTORICAL blocks; the query's own block is force-included
//     and causally masked, and blocks after it are never selected.
//   * The hard top-k is piecewise constant in the parameters, so NO gradient
//     flows through the gate into Q/K. The K gradient buffer is freshly zeroed
//     every backward call so unselected keys stay exactly zero.
//   * Head axis is flattened into tensor ROWS (h*N + t) because the repo's
//     Tensor is 2-D only. A `h, t` vs `h*N+t` slip passes at num_heads==1.
//   * Projection parameter grads go into raw grad_W_* tensors (not via
//     Dense::backward) because the per-head restricted chain needs manual
//     index control. Bias grads use the Dense members' own grad_bias, per repo
//     convention.

#include "moba.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

inline double moba_gelu(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double moba_gelu_deriv(double x) {
    double cdf = 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    return cdf + x * pdf;
}

} // namespace

// ============================================================================
// MoBAAttention
// ============================================================================

MoBAAttention::MoBAAttention(size_t d_model, size_t num_heads,
                             size_t block_size, size_t top_k, bool causal)
    // Guard every member dimension so a 0/0 construction cannot divide by zero
    // in the init list before the validation throws below.
    : W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      W_o(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_q(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_k(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_v(d_model ? d_model : 1, d_model ? d_model : 1),
      grad_W_o(d_model ? d_model : 1, d_model ? d_model : 1),
      d_model_(d_model),
      num_heads_(num_heads),
      head_dim_((num_heads && d_model && d_model % num_heads == 0)
                    ? d_model / num_heads : 1),
      block_size_(block_size),
      top_k_(top_k),
      causal_(causal),
      inv_temp_(1.0 / std::sqrt(static_cast<double>(
          (num_heads && d_model && d_model % num_heads == 0)
              ? d_model / num_heads : 1)))
{
    if (d_model == 0)
        throw std::invalid_argument("MoBAAttention: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("MoBAAttention: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument(
            "MoBAAttention: d_model must be divisible by num_heads");
    if (block_size == 0)
        throw std::invalid_argument("MoBAAttention: block_size must be > 0");
    if (top_k == 0)
        throw std::invalid_argument("MoBAAttention: top_k must be > 0");

    grad_W_q.fill(0.0); grad_W_k.fill(0.0);
    grad_W_v.fill(0.0); grad_W_o.fill(0.0);
    W_q.grad_bias.fill(0.0); W_k.grad_bias.fill(0.0);
    W_v.grad_bias.fill(0.0); W_o.grad_bias.fill(0.0);
}

Tensor MoBAAttention::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument(
            "MoBAAttention: input.cols must equal d_model");
    if (input.rows == 0)
        throw std::invalid_argument("MoBAAttention: input.rows must be > 0");
    if (input.rows % block_size_ != 0)
        throw std::invalid_argument(
            "MoBAAttention: N must be divisible by block_size");

    const size_t N     = input.rows;
    const size_t H     = num_heads_;
    const size_t dh    = head_dim_;
    const size_t n_blk = N / block_size_;
    N_last_     = N;
    n_blk_last_ = n_blk;
    last_input_ = input.clone();

    // Projections. Dense convention: Y = X @ W^T + b, W is (out, in).
    last_Q_ = W_q.forward(input);
    last_K_ = W_k.forward(input);
    last_V_ = W_v.forward(input);
    const Tensor& Q = last_Q_;
    const Tensor& K = last_K_;
    const Tensor& V = last_V_;

    // --- Kbar: mean-pool each contiguous block of keys (Algorithm 1 line 4) ---
    Tensor Kbar(n_blk, d_model_);
    for (size_t b = 0; b < n_blk; ++b)
        for (size_t c = 0; c < d_model_; ++c) {
            double s = 0.0;
            for (size_t t = b * block_size_; t < (b + 1) * block_size_; ++t)
                s += K(t, c);
            Kbar(b, c) = s / static_cast<double>(block_size_);
        }

    // --- S: UNSCALED block affinity, rows flattened by head (Eq. 6) ---
    Tensor S(H * N, n_blk);
    for (size_t h = 0; h < H; ++h) {
        const size_t off = h * dh;
        for (size_t t = 0; t < N; ++t)
            for (size_t b = 0; b < n_blk; ++b) {
                double dot = 0.0;
                for (size_t c = 0; c < dh; ++c)
                    dot += Q(t, off + c) * Kbar(b, off + c);
                S(h * N + t, b) = dot;
            }
    }
    last_scores_ = S.clone();

    // --- causal top-k gate (Eq. 5 + the current-block rule) ---
    // Query t may only select blocks 0..cur-1 (historical); cur is force-in;
    // blocks > cur are future and never selected.
    last_gate_ = Tensor::zeros(H * N, n_blk);
    selected_.assign(H, std::vector<std::vector<size_t>>(N));
    std::vector<size_t> order(n_blk);
    for (size_t h = 0; h < H; ++h)
        for (size_t t = 0; t < N; ++t) {
            const size_t row    = h * N + t;
            const size_t cur    = t / block_size_;
            const size_t n_hist = cur;              // historical blocks 0..cur-1
            for (size_t b = 0; b < n_hist; ++b) order[b] = b;
            std::sort(order.begin(), order.begin() + n_hist,
                      [&](size_t x, size_t y) {
                          if (S(row, x) != S(row, y)) return S(row, x) > S(row, y);
                          return x < y;             // deterministic tie-break
                      });
            const size_t k_sel = std::min(top_k_, n_hist);
            auto& sel = selected_[h][t];
            sel.clear();
            for (size_t m = 0; m < k_sel; ++m) {
                last_gate_(row, order[m]) = 1.0;
                sel.push_back(order[m]);
            }
            last_gate_(row, cur) = 1.0;             // force the current block on
            sel.push_back(cur);
            std::sort(sel.begin(), sel.end());
        }

    // --- one softmax over the union I_t (== the paper's tiled combine) ---
    Tensor A(H * N, N);
    Tensor O(N, d_model_);
    std::vector<size_t> keys;
    std::vector<double>  z, ex;
    for (size_t h = 0; h < H; ++h) {
        const size_t off = h * dh;
        for (size_t t = 0; t < N; ++t) {
            const size_t row = h * N + t;
            // Gather the key indices in I_t. Inside the current block the
            // causal mask truncates to keys <= t; historical blocks are full.
            keys.clear();
            for (size_t b : selected_[h][t]) {
                const size_t lo  = b * block_size_;
                const size_t hi  = lo + block_size_;
                const size_t end = (causal_ && b == t / block_size_)
                                       ? std::min(hi, t + 1) : hi;
                for (size_t i = lo; i < end; ++i) keys.push_back(i);
            }
            z.assign(keys.size(), 0.0);
            double qmax = -std::numeric_limits<double>::infinity();
            for (size_t m = 0; m < keys.size(); ++m) {
                double dot = 0.0;
                for (size_t c = 0; c < dh; ++c)
                    dot += Q(t, off + c) * K(keys[m], off + c);
                z[m] = dot * inv_temp_;
                qmax = std::max(qmax, z[m]);
            }
            ex.assign(keys.size(), 0.0);
            double denom = 0.0;
            for (size_t m = 0; m < keys.size(); ++m) {
                ex[m] = std::exp(z[m] - qmax);
                denom += ex[m];
            }
            for (size_t m = 0; m < keys.size(); ++m)
                A(row, keys[m]) = ex[m] / denom;
            for (size_t c = 0; c < dh; ++c) {
                double acc = 0.0;
                for (size_t m = 0; m < keys.size(); ++m)
                    acc += (ex[m] / denom) * V(keys[m], off + c);
                O(t, off + c) = acc;
            }
        }
    }
    last_A_ = A.clone();
    last_O_ = O.clone();
    return W_o.forward(O);
}

Tensor MoBAAttention::backward(const Tensor& grad_output, double /* lr */) {
    if (grad_output.rows != N_last_ || grad_output.cols != d_model_)
        throw std::invalid_argument(
            "MoBAAttention::backward: grad_output shape must match forward output");
    if (N_last_ == 0)
        throw std::invalid_argument(
            "MoBAAttention::backward: forward must be called first");

    const size_t N     = N_last_;
    const size_t H     = num_heads_;
    const size_t dh    = head_dim_;
    const size_t n_blk = n_blk_last_;

    // Output projection: out = O @ W_o^T + b_o, so
    //   dL/dW_o[j,i] = sum_t grad_output[t,j] * O[t,i]      (grad_output, NOT dO)
    //   dL/db_o[j]   = sum_t grad_output[t,j]
    //   dO[t,i]      = sum_j grad_output[t,j] * W_o[j,i]
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
            const double g = grad_output(t, i);
            bacc += g;
            for (size_t j = 0; j < d_model_; ++j)
                grad_W_o(i, j) += g * last_O_(t, j);
        }
        W_o.grad_bias(0, i) += bacc;
    }

    // Freshly zeroed per-call scratch: unselected keys must stay exactly 0.
    Tensor dQ(N, d_model_), dK(N, d_model_), dV(N, d_model_);
    dQ.fill(0.0); dK.fill(0.0); dV.fill(0.0);

    std::vector<size_t> keys;
    std::vector<double>  dS;
    for (size_t h = 0; h < H; ++h) {
        const size_t off = h * dh;
        for (size_t t = 0; t < N; ++t) {
            const size_t row = h * N + t;
            keys.clear();
            for (size_t b : selected_[h][t]) {
                const size_t lo  = b * block_size_;
                const size_t hi  = lo + block_size_;
                const size_t end = (causal_ && b == t / block_size_)
                                       ? std::min(hi, t + 1) : hi;
                for (size_t i = lo; i < end; ++i) keys.push_back(i);
            }
            dS.assign(keys.size(), 0.0);

            // dA[i,t] = sum_d dO[t,d] * v_i,d  (no inv_temp: A is post-softmax)
            double sum_ad = 0.0;
            for (size_t m = 0; m < keys.size(); ++m) {
                const size_t i = keys[m];
                const double a = last_A_(row, i);
                double dot_v = 0.0;
                for (size_t d = 0; d < dh; ++d) {
                    const double g = dO(t, off + d);
                    dot_v += g * last_V_(i, off + d);
                    dV(i, off + d) += a * g;
                }
                // softmax backward: dS = A * (dA - sum_j A_j dA_j)
                dS[m] = a * dot_v;
                sum_ad += a * dot_v;
            }
            // dz = dS * inv_temp; dq_j += dz * k_i ; dk_i += dz * q_j
            for (size_t m = 0; m < keys.size(); ++m) {
                const size_t i = keys[m];
                const double s = (dS[m] - last_A_(row, i) * sum_ad) * inv_temp_;
                for (size_t d = 0; d < dh; ++d) {
                    dQ(t, off + d) += s * last_K_(i, off + d);
                    dK(i, off + d) += s * last_Q_(t, off + d);
                }
            }
        }
    }
    (void)n_blk;
    last_dK_ = dK.clone();

    // Projection params + input gradient. For Dense W (out, in), Y = X W^T + b:
    //   grad_W += dY^T X ; grad_b += colsum(dY) ; d_input += dY W
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

void MoBAAttention::update_weights(double lr) {
    W_q.weights -= grad_W_q * lr;
    W_k.weights -= grad_W_k * lr;
    W_v.weights -= grad_W_v * lr;
    W_o.weights -= grad_W_o * lr;
    W_q.bias -= W_q.grad_bias * lr;
    W_k.bias -= W_k.grad_bias * lr;
    W_v.bias -= W_v.grad_bias * lr;
    W_o.bias -= W_o.grad_bias * lr;
}

void MoBAAttention::zero_grad() {
    grad_W_q.fill(0.0); W_q.grad_bias.fill(0.0);
    grad_W_k.fill(0.0); W_k.grad_bias.fill(0.0);
    grad_W_v.fill(0.0); W_v.grad_bias.fill(0.0);
    grad_W_o.fill(0.0); W_o.grad_bias.fill(0.0);
}

std::vector<Tensor*> MoBAAttention::parameters() {
    return {&W_q.weights, &W_q.bias,
            &W_k.weights, &W_k.bias,
            &W_v.weights, &W_v.bias,
            &W_o.weights, &W_o.bias};
}

std::vector<Tensor*> MoBAAttention::gradients() {
    return {&grad_W_q, &W_q.grad_bias,
            &grad_W_k, &W_k.grad_bias,
            &grad_W_v, &W_v.grad_bias,
            &grad_W_o, &W_o.grad_bias};
}

// ============================================================================
// MoBABlock
// ============================================================================

MoBABlock::MoBABlock(size_t d_model, size_t num_heads, size_t block_size,
                     size_t top_k, size_t ffn_dim)
    : attn(d_model, num_heads, block_size, top_k),
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
        throw std::invalid_argument("MoBABlock: d_model must be > 0");
}

Tensor MoBABlock::forward(const Tensor& input) {
    last_x_ = input.clone();
    last_z1_ = ln1.forward(input);
    last_attn_out_ = attn.forward(last_z1_);
    last_res1_ = last_z1_ + last_attn_out_;
    last_z2_ = ln2.forward(last_res1_);
    last_h_pre_ = ffn_fc1_.forward(last_z2_);
    last_h_act_ = last_h_pre_.apply(moba_gelu);
    Tensor ffn_out = ffn_fc2_.forward(last_h_act_);
    return last_res1_ + ffn_out;
}

Tensor MoBABlock::backward(const Tensor& grad_output, double lr) {
    // out = res1 + ffn_out  =>  d_res1 = grad_output + (chain through FFN)
    Tensor d_h_act = ffn_fc2_.backward(grad_output, lr);
    Tensor d_h_pre(d_h_act.rows, d_h_act.cols);
    for (size_t i = 0; i < d_h_act.rows; ++i)
        for (size_t j = 0; j < d_h_act.cols; ++j)
            d_h_pre(i, j) = d_h_act(i, j) * moba_gelu_deriv(last_h_pre_(i, j));
    Tensor d_z2 = ffn_fc1_.backward(d_h_pre, lr);

    // z2 = ln2(res1) -> route d_z2 THROUGH ln2, do not add it directly.
    Tensor d_res1 = grad_output + ln2.backward(d_z2, lr);

    // res1 = z1 + attn_out
    Tensor d_z1 = d_res1 + attn.backward(d_res1, lr);

    // z1 = ln1(x)
    return ln1.backward(d_z1, lr);
}

void MoBABlock::update_weights(double lr) {
    attn.update_weights(lr);
    ln1.update_weights(lr);
    ln2.update_weights(lr);
    ffn_fc1_.update_weights(lr);
    ffn_fc2_.update_weights(lr);
}

void MoBABlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

std::vector<Tensor*> MoBABlock::parameters() {
    std::vector<Tensor*> p = attn.parameters();
    for (Tensor* t : ln1.parameters())      p.push_back(t);
    for (Tensor* t : ln2.parameters())      p.push_back(t);
    for (Tensor* t : ffn_fc1_.parameters()) p.push_back(t);
    for (Tensor* t : ffn_fc2_.parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> MoBABlock::gradients() {
    std::vector<Tensor*> g = attn.gradients();
    for (Tensor* t : ln1.gradients())      g.push_back(t);
    for (Tensor* t : ln2.gradients())      g.push_back(t);
    for (Tensor* t : ffn_fc1_.gradients()) g.push_back(t);
    for (Tensor* t : ffn_fc2_.gradients()) g.push_back(t);
    return g;
}

// ============================================================================
// MoBAModel
// ============================================================================

MoBAModel::MoBAModel(size_t input_dim, size_t d_model, size_t output_dim,
                     size_t num_blocks, size_t block_size, size_t top_k,
                     size_t num_heads)
    : input_proj(input_dim ? input_dim : 1, d_model ? d_model : 1),
      final_ln(d_model ? d_model : 1),
      classifier(d_model ? d_model : 1, output_dim ? output_dim : 1),
      input_dim_(input_dim), d_model_(d_model), output_dim_(output_dim),
      num_blocks_(num_blocks)
{
    if (input_dim == 0)
        throw std::invalid_argument("MoBAModel: input_dim must be > 0");
    if (d_model == 0)
        throw std::invalid_argument("MoBAModel: d_model must be > 0");
    if (output_dim == 0)
        throw std::invalid_argument("MoBAModel: output_dim must be > 0");
    if (num_blocks == 0)
        throw std::invalid_argument("MoBAModel: num_blocks must be > 0");
    if (block_size == 0)
        throw std::invalid_argument("MoBAModel: block_size must be > 0");

    // Constructing the first block validates d_model / num_heads / top_k /
    // block_size, so the model inherits those throws rather than duplicating
    // the checks here.
    for (size_t b = 0; b < num_blocks; ++b)
        blocks.push_back(std::make_unique<MoBABlock>(
            d_model, num_heads, block_size, top_k, 0));
}

Tensor MoBAModel::forward(const Tensor& input) {
    Tensor h = input_proj.forward(input);
    for (auto& blk : blocks) h = blk->forward(h);
    h = final_ln.forward(h);
    return classifier.forward(h);
}

Tensor MoBAModel::backward(const Tensor& grad_output, double lr) {
    Tensor g = classifier.backward(grad_output, lr);
    g = final_ln.backward(g, lr);
    for (size_t i = blocks.size(); i > 0; --i)
        g = blocks[i - 1]->backward(g, lr);
    return input_proj.backward(g, lr);
}

void MoBAModel::update_weights(double lr) {
    input_proj.update_weights(lr);
    for (auto& blk : blocks) blk->update_weights(lr);
    final_ln.update_weights(lr);
    classifier.update_weights(lr);
}

void MoBAModel::zero_grad() {
    input_proj.zero_grad();
    for (auto& blk : blocks) blk->zero_grad();
    final_ln.zero_grad();
    classifier.zero_grad();
}

std::vector<Tensor*> MoBAModel::parameters() {
    std::vector<Tensor*> p = input_proj.parameters();
    for (auto& blk : blocks)
        for (Tensor* t : blk->parameters()) p.push_back(t);
    for (Tensor* t : final_ln.parameters())   p.push_back(t);
    for (Tensor* t : classifier.parameters()) p.push_back(t);
    return p;
}

std::vector<Tensor*> MoBAModel::gradients() {
    std::vector<Tensor*> g = input_proj.gradients();
    for (auto& blk : blocks)
        for (Tensor* t : blk->gradients()) g.push_back(t);
    for (Tensor* t : final_ln.gradients())   g.push_back(t);
    for (Tensor* t : classifier.gradients()) g.push_back(t);
    return g;
}