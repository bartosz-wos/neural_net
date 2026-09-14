// ============================================================================
// Expire-Span Attention — Sukhbaatar et al. 2021 implementation
// ============================================================================

#include "expire_span.h"
#include "../../activations/activations.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

inline Tensor row_softmax(const Tensor& x) {
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double row_max = x[i][0];
        for (size_t j = 1; j < x.cols; ++j)
            if (x[i][j] > row_max) row_max = x[i][j];
        double sum = 0.0;
        for (size_t j = 0; j < x.cols; ++j) {
            double e = std::exp(x[i][j] - row_max);
            result[i][j] = e;
            sum += e;
        }
        double inv = 1.0 / (sum + 1e-12);
        for (size_t j = 0; j < x.cols; ++j)
            result[i][j] *= inv;
    }
    return result;
}

inline double gelu_deriv(double x) {
    double xc = std::max(-4.0, std::min(4.0, x));
    double u  = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
    double th = std::tanh(u);
    double du = std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * xc * xc);
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
}

}  // namespace

// ============================================================================
// ExpireSpanAttention
// ============================================================================

ExpireSpanAttention::ExpireSpanAttention(size_t d_model,
                                         size_t num_query_heads,
                                         size_t num_kv_heads,
                                         size_t S_max,
                                         double s_min)
    : W_q(d_model, d_model),
      W_k(d_model, d_model),
      W_v(d_model, d_model),
      W_o(d_model, d_model),
      W_span(1, d_model),
      b_span(1, 1),
      grad_W_q(d_model, d_model),
      grad_W_k(d_model, d_model),
      grad_W_v(d_model, d_model),
      grad_W_o(d_model, d_model),
      grad_W_span(1, d_model),
      grad_b_span(1, 1),
      d_model_(d_model),
      num_query_heads_(num_query_heads),
      num_kv_heads_(num_kv_heads),
      S_max_(S_max),
      s_min_(s_min),
      soft_threshold_(1.0),
      temperature_(1.0)
{
    // Validate first — must happen BEFORE we compute head_dim_ or scale_,
    // both of which would div-by-zero on invalid config.
    if (d_model_ == 0) {
        throw std::invalid_argument("ExpireSpanAttention: d_model must be > 0");
    }
    if (num_query_heads_ == 0) {
        throw std::invalid_argument("ExpireSpanAttention: num_query_heads must be > 0");
    }
    if (num_kv_heads_ == 0) {
        throw std::invalid_argument("ExpireSpanAttention: num_kv_heads must be > 0");
    }
    if (d_model_ % num_query_heads_ != 0) {
        throw std::invalid_argument(
            "ExpireSpanAttention: d_model must be evenly divisible by num_query_heads");
    }
    if (num_query_heads_ % num_kv_heads_ != 0) {
        throw std::invalid_argument(
            "ExpireSpanAttention: num_query_heads must be evenly divisible by num_kv_heads");
    }
    if (s_min_ < 0.0) {
        throw std::invalid_argument("ExpireSpanAttention: s_min must be >= 0");
    }

    head_dim_ = d_model_ / num_query_heads_;
    group_size_ = num_query_heads_ / num_kv_heads_;
    scale_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));

    // Initialize the active K/V head blocks with small random values.
    // (The remaining tail of W_k/W_v beyond num_kv_heads * head_dim stays 0.)
    {
        Tensor W_q_init = Tensor::random(d_model_, d_model_, 0.02);
        Tensor W_o_init = Tensor::random(d_model_, d_model_, 0.02);
        W_q = W_q_init;
        W_o = W_o_init;
        size_t kv_dim = num_kv_heads_ * head_dim_;
        for (size_t i = 0; i < kv_dim; ++i)
            for (size_t j = 0; j < d_model_; ++j) {
                W_k[i][j] = 0.02 * (2.0 * (static_cast<double>((i * 7 + j * 13) % 1000) / 1000.0) - 1.0);
                W_v[i][j] = 0.02 * (2.0 * (static_cast<double>((i * 11 + j * 17) % 1000) / 1000.0) - 1.0);
            }
    }
}

std::vector<Tensor*> ExpireSpanAttention::parameters() {
    return {&W_q, &W_k, &W_v, &W_o, &W_span, &b_span};
}

std::vector<Tensor*> ExpireSpanAttention::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o, &grad_W_span, &grad_b_span};
}

void ExpireSpanAttention::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
    grad_W_span.fill(0.0);
    grad_b_span.fill(0.0);
}

void ExpireSpanAttention::update_weights(double learning_rate) {
    W_q    -= grad_W_q    * learning_rate;
    W_k    -= grad_W_k    * learning_rate;
    W_v    -= grad_W_v    * learning_rate;
    W_o    -= grad_W_o    * learning_rate;
    W_span -= grad_W_span * learning_rate;
    b_span -= grad_b_span * learning_rate;
}

Tensor ExpireSpanAttention::forward(const Tensor& input) {
    const size_t n = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("ExpireSpanAttention.forward: input cols must equal d_model");
    }

    last_input_ = input.clone();

    // Q/K/V projections: (n, d_model) each, no biases.
    Tensor Q(n, d_model_), K(n, d_model_), V(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double qv = 0.0, kv = 0.0, vv = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                qv += input[i][k] * W_q[k][j];
                kv += input[i][k] * W_k[k][j];
                vv += input[i][k] * W_v[k][j];
            }
            Q[i][j] = qv;
            K[i][j] = kv;
            V[i][j] = vv;
        }
    }
    last_q_ = Q;
    last_k_ = K;
    last_v_ = V;

    // Span head: z_i = sum_k W_span[0][k] * x_i[k] + b_span[0][0], shape (n, 1)
    Tensor z(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double s = b_span[0][0];
        for (size_t k = 0; k < d_model_; ++k) s += W_span[0][k] * input[i][k];
        z[i][0] = s;
    }
    last_z_ = z;

    // span = clamp(sigmoid(z), s_min, 1.0)
    Tensor span(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double sig = 1.0 / (1.0 + std::exp(-z[i][0]));
        if (sig < s_min_) sig = s_min_;
        if (sig > 1.0)   sig = 1.0;
        span[i][0] = sig;
    }
    last_span_ = span;

    // Effective max age. If user passed S_max=0, use n (effectively no prun).
    const double S_max_eff = (S_max_ == 0) ? static_cast<double>(n) : static_cast<double>(S_max_);

    // Build the soft mask (last_soft_mask_) and the saturated last_mask_.
    // last_soft_mask_[i,j] = -soft_threshold * sigmoid((i - j - s_j * S_max_eff) / temperature)
    // last_mask_[i,j]      = -1e9 if (i > j AND soft_mask value < -1e9/2) else 0
    //   (i.e. visually equivalent to "j > i: -1e9" for j > i and "drop" for j ≤ i with too-old age.)
    last_soft_mask_ = Tensor(n, n);
    last_mask_ = Tensor(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double v;
            if (j > i) {
                // Causal: always drop.
                v = 0.0; // logit value
                last_soft_mask_[i][j] = -soft_threshold_;
                last_mask_[i][j] = -1e9;
            } else {
                double age = static_cast<double>(i) - static_cast<double>(j);
                double thr = span[j][0] * S_max_eff;
                // Note: arg = (age - threshold) / temperature; positive → drop.
                double arg = (age - thr) / temperature_;
                double sig = 1.0 / (1.0 + std::exp(-arg));
                double soft = -soft_threshold_ * sig;
                last_soft_mask_[i][j] = soft;
                // For the visible mask tensor, saturate to either 0 or -1e9.
                last_mask_[i][j] = (arg > 0.0) ? -1e9 : 0.0;
            }
            (void)v;
        }
    }

    // Per-head attention.
    Tensor head_out(n, d_model_);
    head_out.fill(0.0);
    last_attn_ = Tensor(num_query_heads_ * n, n);
    last_attn_by_head_.clear();
    last_attn_by_head_.reserve(num_query_heads_);

    for (size_t qh = 0; qh < num_query_heads_; ++qh) {
        const size_t kh = qh / group_size_;
        const size_t q_off  = qh * head_dim_;
        const size_t kv_off = kh * head_dim_;

        // scores = (Q_h @ K_h^T) * scale  : (n, n)
        Tensor scores(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (size_t d = 0; d < head_dim_; ++d) {
                    s += Q[i][q_off + d] * K[j][kv_off + d];
                }
                scores[i][j] = s * scale_;
            }
        }

        // Add the soft mask additively: pre_softmax = scores + soft_mask.
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                scores[i][j] += last_soft_mask_[i][j];
            }
        }

        // row softmax → A.
        Tensor A = row_softmax(scores);

        // Cache A per-head for tests + BPTT.
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                last_attn_[qh * n + i][j] = A[i][j];
        last_attn_by_head_.push_back(A);

        // head_out[:, q_off : q_off+head_dim] = A @ V_h  : (n, n) @ (n, head_dim) = (n, head_dim)
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    v += A[i][j] * V[j][kv_off + d];
                }
                head_out[i][q_off + d] = v;
            }
        }
    }

    last_head_out_ = head_out;

    // output = head_out @ W_o : (n, d_model) @ (d_model, d_model) = (n, d_model)
    Tensor output(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double v = 0.0;
            for (size_t k = 0; k < d_model_; ++k) {
                v += head_out[i][k] * W_o[k][j];
            }
            output[i][j] = v;
        }
    }
    return output;
}

Tensor ExpireSpanAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t n = grad_output.rows;
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("ExpireSpanAttention.backward: grad_output cols must equal d_model");
    }

    const double S_max_eff = (S_max_ == 0) ? static_cast<double>(n) : static_cast<double>(S_max_);

    // ---- Step 1: propagate through W_o ----
    // output = head_out @ W_o  →  d_head_out = grad_output @ W_o^T
    Tensor d_head_out(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                v += grad_output[i][j] * W_o[k][j];
            }
            d_head_out[i][k] = v;
        }
    }

    // grad_W_o[k][j] += Σ_t grad_output[t][j] * head_out[t][k]
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < n; ++t) {
                s += grad_output[t][j] * last_head_out_[t][i];
            }
            grad_W_o[i][j] += s;
        }
    }

    // ---- Step 2: per-Q-head backward, with K/V gradient accumulation ----
    // Also accumulate ds_j — gradient of soft_mask w.r.t. span[j] — summed
    // across all surviving (i, j) edges per head. Then we add the per-head
    // contributions together (the soft mask is the same across heads).
    Tensor d_q_acc(n, d_model_);   d_q_acc.fill(0.0);
    Tensor d_k_acc(n, d_model_);   d_k_acc.fill(0.0);
    Tensor d_v_acc(n, d_model_);   d_v_acc.fill(0.0);
    Tensor ds_acc(n, 1);           ds_acc.fill(0.0);   // sum across heads

    for (size_t qh = 0; qh < num_query_heads_; ++qh) {
        const size_t kh = qh / group_size_;
        const size_t q_off  = qh * head_dim_;
        const size_t kv_off = kh * head_dim_;

        // d_V_h = A^T @ d_head_out_h  : (n, n)^T @ (n, head_dim) = (n, head_dim)
        // d_A_h = d_head_out_h @ V_h^T : (n, head_dim) @ (head_dim, n) = (n, n)
        Tensor d_V_h(n, head_dim_);
        Tensor d_A_h(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                double v = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    v += last_attn_[qh * n + j][i] * d_head_out[j][q_off + d];
                }
                d_V_h[i][d] = v;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double v = 0.0;
                for (size_t d = 0; d < head_dim_; ++d) {
                    v += d_head_out[i][q_off + d] * last_v_[j][kv_off + d];
                }
                d_A_h[i][j] = v;
            }
        }

        // Softmax backward: d_pre_soft = A * (d_A - row_sum(d_A * A))
        Tensor d_pre_soft(n, n);
        for (size_t i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                row_sum += last_attn_[qh * n + i][j] * d_A_h[i][j];
            }
            for (size_t j = 0; j < n; ++j) {
                d_pre_soft[i][j] = last_attn_[qh * n + i][j] *
                                   (d_A_h[i][j] - row_sum);
            }
        }

        // d_Q_h = scale_ * (d_pre_soft @ K_h)   (n, head_dim)
        // d_K_h = scale_ * (d_pre_soft^T @ Q_h) (n, head_dim)
        Tensor d_Q_h(n, head_dim_);
        Tensor d_K_h(n, head_dim_);
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                double qv = 0.0, kv = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    qv += d_pre_soft[i][j] * last_k_[j][kv_off + d];
                    kv += d_pre_soft[j][i] * last_q_[j][q_off + d];
                }
                d_Q_h[i][d] = qv * scale_;
                d_K_h[i][d] = kv * scale_;
            }
        }

        // Accumulate.
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < head_dim_; ++d) {
                d_q_acc[i][q_off + d]  += d_Q_h[i][d];
                d_k_acc[i][kv_off + d] += d_K_h[i][d];
                d_v_acc[i][kv_off + d] += d_V_h[i][d];
            }
        }

        // ---- Span-head gradient: ds_j += Σ_i d_pre_soft[i, j] · d_soft_mask[i,j] / ds_j ----
        // Forward: soft_mask[i, j] = -soft_threshold · sigmoid(arg_ij)
        //   arg_ij = (i - j - s_j · S_max_eff) / temperature
        // d/d_s_j arg_ij = -S_max_eff / temperature
        // d/d_arg sigmoid(arg) = sigmoid(arg) · (1 - sigmoid(arg))
        // So: d_soft_mask[i,j] / d_s_j = -soft_threshold · sigmoid'(arg_ij) · (-S_max_eff / temperature)
        //                              = soft_threshold · S_max_eff / temperature · sigmoid(arg) · (1 - sigmoid(arg))
        //
        // For each j, ds_acc[j] += Σ_{i ≥ j} d_pre_soft[i, j] · [the chain above]
        // Note: d_pre_soft[i, j] is naturally 0 on dropped positions (since A[i,j] is 0
        // and the softmax gradient chain kills it). So we just sum over all (i, j)
        // and the dropped contributions are zero automatically.
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {  // j ≤ i (causal); also avoids j > i where mask = -1e9
                double arg = (static_cast<double>(i) - static_cast<double>(j)
                              - last_span_[j][0] * S_max_eff) / temperature_;
                double sig = 1.0 / (1.0 + std::exp(-arg));
                double chain = soft_threshold_ * S_max_eff / temperature_
                             * sig * (1.0 - sig);
                ds_acc[j][0] += d_pre_soft[i][j] * chain;
            }
        }
    }

    // ---- Step 3: parameter gradients from d_q/d_k/d_v vs last_input_ ----
    for (size_t i = 0; i < d_model_; ++i) {
        for (size_t j = 0; j < d_model_; ++j) {
            double sq = 0.0, sk = 0.0, sv = 0.0;
            for (size_t t = 0; t < n; ++t) {
                sq += d_q_acc[t][j] * last_input_[t][i];
                sk += d_k_acc[t][j] * last_input_[t][i];
                sv += d_v_acc[t][j] * last_input_[t][i];
            }
            grad_W_q[i][j] += sq;
            grad_W_k[i][j] += sk;
            grad_W_v[i][j] += sv;
        }
    }

    // ---- Step 4: span-head parameter gradients + input gradient ----
    // dz = ds · sigmoid'(z) = ds · sigmoid(z) · (1 - sigmoid(z))
    //   Note: we use the unclamped sigmoid derivative here. The clamp in forward
    //   is a forward-only stabilization; the gradient flows through the sigmoid.
    // grad_W_span[0][k] += Σ_i dz[i][0] · X[i][k]
    // grad_b_span[0][0] += Σ_i dz[i][0]
    Tensor dz(n, 1);
    for (size_t i = 0; i < n; ++i) {
        double zv = last_z_[i][0];
        double sig = 1.0 / (1.0 + std::exp(-zv));
        dz[i][0] = ds_acc[i][0] * sig * (1.0 - sig);
    }
    for (size_t k = 0; k < d_model_; ++k) {
        double sw = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sw += dz[i][0] * last_input_[i][k];
        }
        grad_W_span[0][k] += sw;
    }
    double sb = 0.0;
    for (size_t i = 0; i < n; ++i) sb += dz[i][0];
    grad_b_span[0][0] += sb;

    // ---- Step 5: full input gradient ----
    // d_input = d_q @ W_q^T + d_k @ W_k^T + d_v @ W_v^T + dz @ W_span
    Tensor d_input(n, d_model_);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < d_model_; ++k) {
            double v = 0.0;
            for (size_t j = 0; j < d_model_; ++j) {
                v += d_q_acc[i][j] * W_q[k][j]
                   + d_k_acc[i][j] * W_k[k][j]
                   + d_v_acc[i][j] * W_v[k][j];
            }
            // span head contribution: d_input[i][k] += dz[i][0] * W_span[0][k]
            v += dz[i][0] * W_span[0][k];
            d_input[i][k] = v;
        }
    }
    last_d_input_ = d_input.clone();
    return d_input;
}

// ============================================================================
// ExpireSpanBlock — stub (Tasks 6+)
// ============================================================================

ExpireSpanBlock::ExpireSpanBlock(size_t d_model,
                                 size_t num_query_heads,
                                 size_t num_kv_heads,
                                 size_t S_max,
                                 double s_min,
                                 size_t ffn_dim)
    : d_model_(d_model),
      ffn_dim_(ffn_dim == 0 ? 4 * d_model : ffn_dim),
      attn_(d_model, num_query_heads, num_kv_heads, S_max, s_min),
      ln1_(d_model),
      ln2_(d_model),
      ffn_fc1_(d_model, ffn_dim_),
      ffn_fc2_(ffn_dim_, d_model)
{
}

std::vector<Tensor*> ExpireSpanBlock::parameters() {
    auto p = attn_.parameters();
    auto p1 = ln1_.parameters();
    auto p2 = ln2_.parameters();
    auto pf1 = ffn_fc1_.parameters();
    auto pf2 = ffn_fc2_.parameters();
    p.insert(p.end(), p1.begin(), p1.end());
    p.insert(p.end(), p2.begin(), p2.end());
    p.insert(p.end(), pf1.begin(), pf1.end());
    p.insert(p.end(), pf2.begin(), pf2.end());
    return p;
}

std::vector<Tensor*> ExpireSpanBlock::gradients() {
    auto g = attn_.gradients();
    auto g1 = ln1_.gradients();
    auto g2 = ln2_.gradients();
    auto gf1 = ffn_fc1_.gradients();
    auto gf2 = ffn_fc2_.gradients();
    g.insert(g.end(), g1.begin(), g1.end());
    g.insert(g.end(), g2.begin(), g2.end());
    g.insert(g.end(), gf1.begin(), gf1.end());
    g.insert(g.end(), gf2.begin(), gf2.end());
    return g;
}

void ExpireSpanBlock::zero_grad() {
    attn_.zero_grad();
    ln1_.zero_grad();
    ln2_.zero_grad();
    ffn_fc1_.zero_grad();
    ffn_fc2_.zero_grad();
}

void ExpireSpanBlock::update_weights(double learning_rate) {
    attn_.update_weights(learning_rate);
    ln1_.update_weights(learning_rate);
    ln2_.update_weights(learning_rate);
    ffn_fc1_.update_weights(learning_rate);
    ffn_fc2_.update_weights(learning_rate);
}

Tensor ExpireSpanBlock::forward(const Tensor& input) {
    const size_t n = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("ExpireSpanBlock.forward: input cols must equal d_model");
    }

    last_input_ = input.clone();

    // Pre-LN → attention → residual.
    last_z1_ = ln1_.forward(input);
    last_attn_out_ = attn_.forward(last_z1_);
    last_res1_ = Tensor(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            last_res1_[i][j] = input[i][j] + last_attn_out_[i][j];

    if (ffn_dim_ == 0) {
        // No FFN sub-layer — the block is pure attention.
        return last_res1_.clone();
    }

    // Pre-LN → GELU FFN → residual.
    last_z2_ = ln2_.forward(last_res1_);
    last_ffn_pre_ = ffn_fc1_.forward(last_z2_);
    Tensor ffn_act = last_ffn_pre_.apply([](double x) {
        // GELU tanh-approximation matching activations.h convention.
        double xc = std::max(-4.0, std::min(4.0, x));
        double u  = std::sqrt(2.0 / M_PI) * (xc + 0.044715 * xc * xc * xc);
        return 0.5 * xc * (1.0 + std::tanh(u));
    });
    Tensor ffn_out = ffn_fc2_.forward(ffn_act);

    Tensor output(n, d_model_);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            output[i][j] = last_res1_[i][j] + ffn_out[i][j];
    return output;
}

Tensor ExpireSpanBlock::backward(const Tensor& grad_output, double learning_rate) {
    const size_t n = grad_output.rows;
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("ExpireSpanBlock.backward: grad_output cols must equal d_model");
    }

    // Standard transformer-block backward (matches GQABlock convention):
    //   z1      = ln1(input)
    //   attn_o  = attn(z1)
    //   res1    = input + attn_o
    //   z2      = ln2(res1)               [if ffn_dim > 0]
    //   ffn_pre = fc1(z2)                 [if ffn_dim > 0]
    //   ffn_h   = GELU(ffn_pre)           [if ffn_dim > 0]
    //   ffn_o   = fc2(ffn_h)              [if ffn_dim > 0]
    //   output  = res1 + ffn_o            [or output = res1 if ffn_dim == 0]

    // Step 1: residual split at the top.
    Tensor d_res1 = grad_output.clone();
    if (ffn_dim_ > 0) {
        Tensor d_ffn_o = grad_output.clone();

        // Step 2: fc2 backward → d_ffn_h
        Tensor d_ffn_h = ffn_fc2_.backward(d_ffn_o, learning_rate);

        // Step 3: GELU backward (recompute pre-GELU from cached ln2).
        Tensor ffn_pre_recomp = ffn_fc1_.forward(last_z2_);
        Tensor d_ffn_pre(ffn_pre_recomp.rows, ffn_pre_recomp.cols);
        for (size_t i = 0; i < ffn_pre_recomp.rows; ++i)
            for (size_t j = 0; j < ffn_pre_recomp.cols; ++j)
                d_ffn_pre[i][j] = d_ffn_h[i][j] * gelu_deriv(ffn_pre_recomp[i][j]);

        // Step 4: fc1 backward → d_z2
        Tensor d_z2 = ffn_fc1_.backward(d_ffn_pre, learning_rate);

        // Step 5: ln2 backward, accumulating into d_res1.
        Tensor d_res1_from_ln2 = ln2_.backward(d_z2, learning_rate);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d_model_; ++j)
                d_res1[i][j] += d_res1_from_ln2[i][j];
    }

    // Step 6: residual split at res1 = input + attn_o.
    Tensor d_attn_o = d_res1.clone();

    // Step 7: attn backward → d_z1.
    Tensor d_z1 = attn_.backward(d_attn_o, learning_rate);

    // Step 8: ln1 backward + residual add.
    // d_input = ln1.backward(d_z1) + d_res1 (residual carries through).
    Tensor d_input = ln1_.backward(d_z1, learning_rate);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            d_input[i][j] += d_res1[i][j];

    last_d_input_ = d_input.clone();
    return d_input;
}

// ============================================================================
// ExpireSpanModel — stub (Tasks 6+)
// ============================================================================

ExpireSpanModel::ExpireSpanModel(size_t d_input,
                                 size_t d_model,
                                 size_t d_output,
                                 size_t num_blocks,
                                 size_t num_query_heads,
                                 size_t num_kv_heads,
                                 size_t S_max,
                                 double s_min,
                                 size_t ffn_dim)
    : d_input_(d_input),
      d_model_(d_model),
      d_output_(d_output),
      W_in_(d_input, d_model),
      W_out_(d_model, d_output),
      blocks_()
{
    W_in_.init_weights("xavier");
    W_out_.init_weights("xavier");
    for (size_t b = 0; b < num_blocks; ++b) {
        blocks_.emplace_back(d_model, num_query_heads, num_kv_heads, S_max, s_min, ffn_dim);
    }
}

std::vector<Tensor*> ExpireSpanModel::parameters() {
    std::vector<Tensor*> p;
    auto wi = W_in_.parameters();
    auto wo = W_out_.parameters();
    p.insert(p.end(), wi.begin(), wi.end());
    p.insert(p.end(), wo.begin(), wo.end());
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    return p;
}

std::vector<Tensor*> ExpireSpanModel::gradients() {
    std::vector<Tensor*> g;
    auto wi = W_in_.gradients();
    auto wo = W_out_.gradients();
    g.insert(g.end(), wi.begin(), wi.end());
    g.insert(g.end(), wo.begin(), wo.end());
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    return g;
}

void ExpireSpanModel::zero_grad() {
    W_in_.zero_grad();
    W_out_.zero_grad();
    for (auto& b : blocks_) b.zero_grad();
}

void ExpireSpanModel::update_weights(double learning_rate) {
    W_in_.update_weights(learning_rate);
    W_out_.update_weights(learning_rate);
    for (auto& b : blocks_) b.update_weights(learning_rate);
}

Tensor ExpireSpanModel::forward(const Tensor& input) {
    last_input_ = input.clone();
    Tensor x = W_in_.forward(input);
    for (auto& b : blocks_) {
        x = b.forward(x);
    }
    return W_out_.forward(x);
}

Tensor ExpireSpanModel::backward(const Tensor& grad_output, double learning_rate) {
    Tensor d = W_out_.backward(grad_output, learning_rate);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        d = it->backward(d, learning_rate);
    }
    return W_in_.backward(d, learning_rate);
}
