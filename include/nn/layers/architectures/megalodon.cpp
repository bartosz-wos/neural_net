#include "megalodon.h"
#include "../../activations/activations.h"
#include <algorithm>
#include <cmath>

// ============================================================================
// Megalodon — implementation.
//
// Backward chain (reverse order):
//
//   Y = ffn_down(ffn(ln_out)) + x
//     d_ffn_down, d_ffn <- Down.backward, d_ln_out <- d_ffn_out
//   ln_out = ffn_ln(y_hat)
//     d_y_hat <- ln.backward(d_ln_out)              ★ TWO-HOP: grad_x also +1
//   y_hat = h_out + x
//     d_h_out += d_y_hat, and grad_x += d_y_hat
//   h_out = W_o · h + b_o
//     d_W_o, d_h <- W_o.backward
//   h = concat_h(Σ_s A[t,s] · V[s,h]) = Σ_s A ⊙ V
//     dA, dV <- attention
//   A = softmax(masked(Q K^T / sqrt(d_h)))
//     dS <- softmax backward
//   V = SiLU(v_pre); v_pre = W_v · tsn + b_v
//     d_v_pre, d_W_v, d_tsn
//   Q = q_gamma ⊙ z_norm + q_beta ; same for K
//     d_q_gamma, d_q_beta, d_z_norm
//   z_norm = z_pre / rms   (per head)
//     d_z_pre, d_rms
//   z_pre = W_z · cema + b_z
//     d_W_z, d_cema
//   cema = ComplexEMA(tsn)
//     d_tsn <- ComplexEMA.backward
//   tsn = TimestepNorm(x)
//     d_x <- TimestepNorm.backward
//
// The ★ line is the two-hop residual of §3.4: grad_x receives a contribution
// from the attention residual AND a second one from the FFN residual. Omitting
// the second is a clean ~2x error that finite differences catch immediately.
// ============================================================================

namespace {

inline double sigmoid_fn(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

}  // namespace

// ============================================================================
// Constructor
// ============================================================================
MegalodonBlock::MegalodonBlock(size_t d_model, size_t num_heads, size_t ffn_mult,
                               size_t num_groups, size_t cema_ndim, size_t chunk_size)
    : tsn_(d_model, num_groups),
      cema_(d_model, cema_ndim),
      W_z_(d_model, d_model),
      W_v_(d_model, d_model),
      W_o_(d_model, d_model),
      ffn_ln_(d_model),
      ffn_(d_model, (ffn_mult == 0 ? d_model : ffn_mult) * d_model),
      ffn_down_((ffn_mult == 0 ? d_model : ffn_mult) * d_model, d_model),
      q_gamma_(Tensor::zeros(1, d_model)),
      q_beta_(Tensor::zeros(1, d_model)),
      k_gamma_(Tensor::zeros(1, d_model)),
      k_beta_(Tensor::zeros(1, d_model)),
      grad_q_gamma_(Tensor::zeros(1, d_model)),
      grad_q_beta_(Tensor::zeros(1, d_model)),
      grad_k_gamma_(Tensor::zeros(1, d_model)),
      grad_k_beta_(Tensor::zeros(1, d_model)),
      d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(num_heads == 0 ? 0 : d_model / num_heads),
      z_dim_(d_model),
      v_dim_(d_model),
      ffn_mult_(ffn_mult),
      ffn_hidden_((ffn_mult == 0 ? d_model : ffn_mult) * d_model),
      num_groups_(num_groups),
      cema_ndim_(cema_ndim),
      chunk_size_(chunk_size) {
    // NOTE: the sublayer constructors are in the member-init list, so they run
    // BEFORE the body. The d_model==0 / num_heads==0 checks therefore have to
    // guard the sublayer construction itself — a TimestepNorm(0) or
    // ComplexEMA(0) would allocate a zero-size tensor first. Those two throw
    // on their own, which satisfies the contract; the checks below cover the
    // arguments this class owns.
    if (d_model == 0)
        throw std::invalid_argument("MegalodonBlock: d_model must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("MegalodonBlock: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("MegalodonBlock: d_model must be divisible by num_heads");
    if (ffn_mult == 0)
        throw std::invalid_argument("MegalodonBlock: ffn_mult must be > 0");
    if (cema_ndim == 0)
        throw std::invalid_argument("MegalodonBlock: cema_ndim must be > 0");
    if (num_groups > d_model)
        throw std::invalid_argument("MegalodonBlock: num_groups must be <= d_model");

    // Q/K affine on the normalized z stream. The reference initializes these
    // to 1 (they are a temperature, not a gate) — kappa in Eq. 6.
    q_gamma_.fill(1.0);
    k_gamma_.fill(1.0);
    q_beta_.fill(0.0);
    k_beta_.fill(0.0);
}

// ============================================================================
// Forward
// ============================================================================
Tensor MegalodonBlock::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("MegalodonBlock::forward: input.cols != d_model_");
    if (input.rows == 0)
        throw std::invalid_argument("MegalodonBlock::forward: sequence length must be > 0");

    const size_t n = input.rows;
    T_ = n;

    last_input = input.clone();

    // §3.2 — causal cumulative normalization
    last_tsn = tsn_.forward(last_input);

    // §3.1 — complex exponential moving average
    last_cema = cema_.forward(last_tsn);

    // z projection (§3.3)
    last_z_pre = W_z_.forward(last_cema);   // (n, z_dim)

    // Per-head RMS normalization: z' = z / rms_h, rms_h = sqrt(mean(z_h^2) + eps)
    last_rms = Tensor(n, num_heads_);
    last_z_norm = Tensor(n, z_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t h = 0; h < num_heads_; ++h) {
            double ss = 0.0;
            const size_t base = h * head_dim_;
            for (size_t j = 0; j < head_dim_; ++j) {
                const double v = last_z_pre(t, base + j);
                ss += v * v;
            }
            const double r = std::sqrt(ss / static_cast<double>(head_dim_) + rms_eps_);
            last_rms(t, h) = 1.0 / r;
            for (size_t j = 0; j < head_dim_; ++j)
                last_z_norm(t, base + j) = last_z_pre(t, base + j) / r;
        }
    }

    // Q/K affine (§3.3): Q = kappa_q ⊙ z' + mu_q
    last_q = Tensor(n, z_dim_);
    last_k = Tensor(n, z_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < z_dim_; ++j) {
            last_q(t, j) = q_gamma_(0, j) * last_z_norm(t, j) + q_beta_(0, j);
            last_k(t, j) = k_gamma_(0, j) * last_z_norm(t, j) + k_beta_(0, j);
        }
    }

    // V from the timestep-normalized input (reference: v = SiLU(wv(out_tsn)))
    last_v_pre = W_v_.forward(last_tsn);    // (n, v_dim)
    last_v = last_v_pre.apply(Swish());

    // Masked per-head attention scores.
    //
    // last_score_ is (n, H*n): the per-head score for query t, head h, key s
    // lives at column h*n + s. It MUST stay per-head. Collapsing the heads
    // into a single (n, n) tensor of summed scores looks harmless (a single
    // head reads the same) but silently makes every head share ONE attention
    // distribution — forward would then disagree with the (correctly
    // per-head) backward as soon as H > 1, which is what an FD check on
    // d_input catches.
    const double inv_sqrt_head = 1.0 / std::sqrt(static_cast<double>(head_dim_));
    last_score_ = Tensor::zeros(n, num_heads_ * n);
    for (size_t t = 0; t < n; ++t) {
        for (size_t s = 0; s <= t; ++s) {
            if (!visible(t, s)) continue;
            for (size_t h = 0; h < num_heads_; ++h) {
                double dot = 0.0;
                const size_t base = h * head_dim_;
                for (size_t j = 0; j < head_dim_; ++j)
                    dot += last_q(t, base + j) * last_k(s, base + j);
                last_score_(t, h * n + s) = dot * inv_sqrt_head;
            }
        }
    }

    // Per-head softmax over the visible keys, then A · V.
    last_attn = Tensor::zeros(n, n);
    Tensor h_out(n, v_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t base = h * head_dim_;

            // Numerical stability: subtract this head's row max.
            double rowmax = -1e300;
            for (size_t s = 0; s <= t; ++s)
                if (visible(t, s))
                    rowmax = std::max(rowmax, last_score_(t, h * n + s));

            double denom = 0.0;
            for (size_t s = 0; s <= t; ++s) {
                if (!visible(t, s)) continue;
                denom += std::exp(last_score_(t, h * n + s) - rowmax);
            }
            for (size_t j = 0; j < head_dim_; ++j) {
                double acc = 0.0;
                for (size_t s = 0; s <= t; ++s) {
                    if (!visible(t, s)) continue;
                    acc += (std::exp(last_score_(t, h * n + s) - rowmax) / denom)
                         * last_v(s, base + j);
                }
                h_out(t, base + j) = acc;
            }
        }
    }

    // Mean over heads of the attention distribution — the debug/test view
    // only. Nothing downstream of here reads it.
    for (size_t t = 0; t < n; ++t) {
        for (size_t s = 0; s <= t; ++s) {
            if (!visible(t, s)) continue;
            double acc = 0.0;
            for (size_t h = 0; h < num_heads_; ++h) {
                double hmax = -1e300;
                for (size_t u = 0; u <= t; ++u)
                    if (visible(t, u)) hmax = std::max(hmax, last_score_(t, h * n + u));
                double d = 0.0;
                for (size_t u = 0; u <= t; ++u)
                    if (visible(t, u)) d += std::exp(last_score_(t, h * n + u) - hmax);
                acc += std::exp(last_score_(t, h * n + s) - hmax) / d;
            }
            last_attn(t, s) = acc / static_cast<double>(num_heads_);
        }
    }

    // Output projection (§3.3)
    last_h = h_out;
    last_attn_out = W_o_.forward(last_h);   // (n, d_model)

    // Attention residual: Ŷ = h_out_proj + x
    last_y_hat = Tensor(n, d_model_);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d_model_; ++j)
            last_y_hat(t, j) = last_attn_out(t, j) + last_input(t, j);

    // FFN sublayer with plain LayerNorm
    last_ln_out = ffn_ln_.forward(last_y_hat);
    Tensor ffn_hidden = ffn_.forward(last_ln_out);
    last_ffn_out = ffn_down_.forward(ffn_hidden);

    // TWO-HOP residual (§3.4): the FFN residual reuses the block INPUT x,
    // not the attention output Ŷ. This is the paper's variance fix and is
    // the single easiest thing to get wrong here.
    Tensor output(n, d_model_);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d_model_; ++j)
            output(t, j) = last_ffn_out(t, j) + last_input(t, j);
    return output;
}

// ============================================================================
// Backward
// ============================================================================
Tensor MegalodonBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (T_ == 0)
        throw std::logic_error("MegalodonBlock::backward called before forward");
    const size_t n = T_;
    if (grad_output.rows != n || grad_output.cols != d_model_)
        throw std::invalid_argument("MegalodonBlock::backward: grad_output shape mismatch");

    // ---- Reset accumulated parameter gradients -------------------------
    W_z_.grad_weights = Tensor::zeros(W_z_.weights.rows, W_z_.weights.cols);
    W_z_.grad_bias = Tensor::zeros(W_z_.bias.rows, W_z_.bias.cols);
    W_v_.grad_weights = Tensor::zeros(W_v_.weights.rows, W_v_.weights.cols);
    W_v_.grad_bias = Tensor::zeros(W_v_.bias.rows, W_v_.bias.cols);
    W_o_.grad_weights = Tensor::zeros(W_o_.weights.rows, W_o_.weights.cols);
    W_o_.grad_bias = Tensor::zeros(W_o_.bias.rows, W_o_.bias.cols);
    ffn_down_.grad_weights = Tensor::zeros(ffn_down_.weights.rows, ffn_down_.weights.cols);
    ffn_down_.grad_bias = Tensor::zeros(ffn_down_.bias.rows, ffn_down_.bias.cols);
    grad_q_gamma_ = Tensor::zeros(1, z_dim_);
    grad_q_beta_ = Tensor::zeros(1, z_dim_);
    grad_k_gamma_ = Tensor::zeros(1, z_dim_);
    grad_k_beta_ = Tensor::zeros(1, z_dim_);
    tsn_.zero_grad();
    cema_.zero_grad();
    ffn_ln_.zero_grad();
    ffn_.zero_grad();

    // ---- FFN down-projection -------------------------------------------
    Tensor d_ffn_hidden = ffn_down_.backward(grad_output, 0.0);
    Tensor d_ln_out = ffn_.backward(d_ffn_hidden, 0.0);

    // ---- LayerNorm before the FFN --------------------------------------
    Tensor d_y_hat = ffn_ln_.backward(d_ln_out, 0.0);

    // ---- TWO-HOP residual: grad_x gets the FFN residual contribution ---
    grad_x_ = grad_output.clone();
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d_model_; ++j)
            grad_x_(t, j) += d_y_hat(t, j);

    // ---- Attention residual: Ŷ = attn_out + x --------------------------
    Tensor d_attn_out = d_y_hat.clone();

    // ---- Output projection ---------------------------------------------
    Tensor d_h = W_o_.backward(d_attn_out, 0.0);

    // ---- Attention: h = concat_h(Σ_s A_h[t,s] · V[s,h]) ----------------
    // dA for each head, and dV accumulated over queries.
    Tensor d_v = Tensor::zeros(n, v_dim_);
    const double inv_sqrt_head = 1.0 / std::sqrt(static_cast<double>(head_dim_));
    Tensor d_z_norm(n, z_dim_);
    Tensor d_q(n, z_dim_), d_k(n, z_dim_);

    for (size_t h = 0; h < num_heads_; ++h) {
        const size_t base = h * head_dim_;

        for (size_t t = 0; t < n; ++t) {
            // Read the SAME per-head scores the forward used. Recomputing
            // them from last_q/last_k here is mathematically identical but
            // invites exactly the forward/backward divergence this cache
            // exists to prevent.
            double rowmax = -1e300;
            for (size_t s = 0; s <= t; ++s)
                if (visible(t, s))
                    rowmax = std::max(rowmax, last_score_(t, h * n + s));

            double denom = 0.0;
            for (size_t s = 0; s <= t; ++s) {
                if (!visible(t, s)) continue;
                denom += std::exp(last_score_(t, h * n + s) - rowmax);
            }

            // Pass 1: sum_a_da (the A-WEIGHTED row sum, not the raw Σ dA —
            // using the raw sum is the classic softmax-backward bug) and dV.
            double sum_a_da = 0.0;
            for (size_t s = 0; s <= t; ++s) {
                if (!visible(t, s)) continue;
                const double a = std::exp(last_score_(t, h * n + s) - rowmax) / denom;

                // dA[t,s] = Σ_j d_h[t, base+j] · V[s, base+j]
                double dA = 0.0;
                for (size_t j = 0; j < head_dim_; ++j)
                    dA += d_h(t, base + j) * last_v(s, base + j);
                sum_a_da += a * dA;

                // dV[s, base+j] += A[t,s] · d_h[t, base+j]
                for (size_t j = 0; j < head_dim_; ++j)
                    d_v(s, base + j) += a * d_h(t, base + j);
            }

            // Pass 2: dS from the softmax Jacobian, then dQ and dK.
            for (size_t s = 0; s <= t; ++s) {
                if (!visible(t, s)) continue;
                const double a = std::exp(last_score_(t, h * n + s) - rowmax) / denom;
                double dA = 0.0;
                for (size_t j = 0; j < head_dim_; ++j)
                    dA += d_h(t, base + j) * last_v(s, base + j);

                // dS = A · (dA - sum_a_da);  S = (Q·K^T) / sqrt(d_h)
                const double dS = a * (dA - sum_a_da) * inv_sqrt_head;

                for (size_t j = 0; j < head_dim_; ++j) {
                    d_q(t, base + j) += dS * last_k(s, base + j);
                    d_k(s, base + j) += dS * last_q(t, base + j);
                }
            }
        }
    }

    // ---- Q/K affine ---------------------------------------------------
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < z_dim_; ++j) {
            grad_q_gamma_(0, j) += d_q(t, j) * last_z_norm(t, j);
            grad_q_beta_(0, j)  += d_q(t, j);
            grad_k_gamma_(0, j) += d_k(t, j) * last_z_norm(t, j);
            grad_k_beta_(0, j)  += d_k(t, j);
            d_z_norm(t, j) = d_q(t, j) * q_gamma_(0, j) + d_k(t, j) * k_gamma_(0, j);
        }
    }

    // ---- Per-head RMS normalization backward --------------------------
    // Forward:  r = 1/sqrt(ss/d + eps),  ss = SUM_j z_j^2,  z_norm_j = z_j * r
    //
    // The Jacobian is FULL, not diagonal:
    //   dz_norm_j / dz_m = delta_{jm} * r  -  z_j * z_m * r^3 / d
    // Contracting with the incoming gradient g gives
    //   dz_m = SUM_j g_j * dz_norm_j / dz_m
    //        = r * g_m  -  (r^3/d) * z_m * SUM_j g_j * z_j
    //
    // The second term is a CROSS term: z_m is coupled to every other z_j in
    // the same head through the shared rms. Dropping it (i.e. assuming
    // SUM_j g_j z_j == g_m z_m) is the bug this comment exists to prevent —
    // it leaves the CEMA/W_z path gradient flat-wrong at ~5% relative error
    // that finite differences catch but that no shape assertion would.
    Tensor d_z_pre(n, z_dim_);
    for (size_t t = 0; t < n; ++t) {
        for (size_t h = 0; h < num_heads_; ++h) {
            const size_t base = h * head_dim_;
            const double r = last_rms(t, h);

            // SUM_j g_j * z_j over this head — shared by every component m.
            double gz = 0.0;
            for (size_t j = 0; j < head_dim_; ++j)
                gz += d_z_norm(t, base + j) * last_z_pre(t, base + j);

            const double cross = r * r * r / static_cast<double>(head_dim_);
            for (size_t j = 0; j < head_dim_; ++j) {
                d_z_pre(t, base + j) = r * d_z_norm(t, base + j)
                                     - cross * last_z_pre(t, base + j) * gz;
            }
        }
    }

    // ---- z projection --------------------------------------------------
    Tensor d_cema = W_z_.backward(d_z_pre, 0.0);

    // ---- V path --------------------------------------------------------
    Tensor d_v_pre(n, v_dim_);
    for (size_t i = 0; i < d_v.data.size(); ++i) {
        // swish'(u) = swish(u) + sigmoid(u) · (1 - swish(u))
        const double u = last_v_pre.data[i];
        const double sw = last_v.data[i];
        const double sg = sigmoid_fn(u);
        d_v_pre.data[i] = d_v.data[i] * (sw + sg * (1.0 - sw));
    }
    Tensor d_tsn_v = W_v_.backward(d_v_pre, 0.0);

    // ---- CEMA backward -------------------------------------------------
    Tensor d_tsn_cema = cema_.backward(d_cema, 0.0);

    // Sum the two paths that both reach the timestep-normalized input.
    Tensor d_tsn(n, d_model_);
    for (size_t i = 0; i < d_tsn.data.size(); ++i)
        d_tsn.data[i] = d_tsn_v.data[i] + d_tsn_cema.data[i];

    // ---- TimestepNorm backward -----------------------------------------
    Tensor d_x = tsn_.backward(d_tsn, 0.0);

    // ---- Add to the residual gradient ----------------------------------
    for (size_t i = 0; i < grad_x_.data.size(); ++i)
        grad_x_.data[i] += d_x.data[i];

    return grad_x_.clone();
}

// ============================================================================
// update_weights / zero_grad
// ============================================================================
void MegalodonBlock::update_weights(double lr) {
    W_z_.update_weights(lr);
    W_v_.update_weights(lr);
    W_o_.update_weights(lr);
    ffn_down_.update_weights(lr);
    ffn_ln_.update_weights(lr);
    ffn_.update_weights(lr);
    tsn_.update_weights(lr);
    cema_.update_weights(lr);
    q_gamma_ -= grad_q_gamma_ * lr;
    q_beta_  -= grad_q_beta_ * lr;
    k_gamma_ -= grad_k_gamma_ * lr;
    k_beta_  -= grad_k_beta_ * lr;
}

void MegalodonBlock::zero_grad() {
    W_z_.zero_grad();
    W_v_.zero_grad();
    W_o_.zero_grad();
    ffn_down_.zero_grad();
    ffn_ln_.zero_grad();
    ffn_.zero_grad();
    tsn_.zero_grad();
    cema_.zero_grad();
    grad_q_gamma_ = Tensor::zeros(1, z_dim_);
    grad_q_beta_ = Tensor::zeros(1, z_dim_);
    grad_k_gamma_ = Tensor::zeros(1, z_dim_);
    grad_k_beta_ = Tensor::zeros(1, z_dim_);
}

std::vector<Tensor*> MegalodonBlock::parameters() {
    std::vector<Tensor*> p;
    p.push_back(&tsn_.gamma);
    p.push_back(&tsn_.beta);
    p.push_back(&cema_.alpha_);
    p.push_back(&cema_.delta_);
    p.push_back(&cema_.theta_);
    p.push_back(&cema_.eta_re_);
    p.push_back(&cema_.eta_im_);
    p.push_back(&cema_.omega_);
    p.push_back(&W_z_.weights);
    p.push_back(&W_z_.bias);
    p.push_back(&q_gamma_);
    p.push_back(&q_beta_);
    p.push_back(&k_gamma_);
    p.push_back(&k_beta_);
    p.push_back(&W_v_.weights);
    p.push_back(&W_v_.bias);
    p.push_back(&W_o_.weights);
    p.push_back(&W_o_.bias);
    p.push_back(&ffn_ln_.gamma);
    p.push_back(&ffn_ln_.beta);
    p.push_back(&ffn_.w1_weights());
    p.push_back(&ffn_.w1_bias());
    p.push_back(&ffn_.w2_weights());
    p.push_back(&ffn_.w2_bias());
    p.push_back(&ffn_down_.weights);
    p.push_back(&ffn_down_.bias);
    return p;
}

std::vector<Tensor*> MegalodonBlock::gradients() {
    std::vector<Tensor*> g;
    g.push_back(&tsn_.grad_gamma_);
    g.push_back(&tsn_.grad_beta_);
    g.push_back(&cema_.grad_alpha_);
    g.push_back(&cema_.grad_delta_);
    g.push_back(&cema_.grad_theta_);
    g.push_back(&cema_.grad_eta_re_);
    g.push_back(&cema_.grad_eta_im_);
    g.push_back(&cema_.grad_omega_);
    g.push_back(&W_z_.grad_weights);
    g.push_back(&W_z_.grad_bias);
    g.push_back(&grad_q_gamma_);
    g.push_back(&grad_q_beta_);
    g.push_back(&grad_k_gamma_);
    g.push_back(&grad_k_beta_);
    g.push_back(&W_v_.grad_weights);
    g.push_back(&W_v_.grad_bias);
    g.push_back(&W_o_.grad_weights);
    g.push_back(&W_o_.grad_bias);
    g.push_back(&ffn_ln_.grad_gamma_);
    g.push_back(&ffn_ln_.grad_beta_);
    g.push_back(&ffn_.grad_w1_weights());
    g.push_back(&ffn_.grad_w1_bias());
    g.push_back(&ffn_.grad_w2_weights());
    g.push_back(&ffn_.grad_w2_bias());
    g.push_back(&ffn_down_.grad_weights);
    g.push_back(&ffn_down_.grad_bias);
    return g;
}

void MegalodonBlock::copy_params_from(const MegalodonBlock& other) {
    // chunk_size is deliberately NOT part of this check: it is a forward-time
    // mask predicate, not a parameter. Two blocks that differ only in
    // chunk_size hold identical parameters and can share weights — that is
    // how the Megalodon-chunk signature is tested (same weights, different
    // attention masks). Every other config field DOES change a tensor shape.
    if (other.d_model_ != d_model_ || other.num_heads_ != num_heads_ ||
        other.ffn_mult_ != ffn_mult_ || other.num_groups_ != num_groups_ ||
        other.cema_ndim_ != cema_ndim_) {
        throw std::invalid_argument("MegalodonBlock::copy_params_from: configuration mismatch");
    }
    auto p = parameters();
    // `other` is const, so build the source list from a non-const copy of the
    // pointer list. The tensors themselves are only read.
    MegalodonBlock& src = const_cast<MegalodonBlock&>(other);
    auto op = src.parameters();
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i]->rows != op[i]->rows || p[i]->cols != op[i]->cols)
            throw std::invalid_argument("MegalodonBlock::copy_params_from: tensor shape mismatch");
        *p[i] = op[i]->clone();
    }
}

// ============================================================================
// MegalodonModel
// ============================================================================
MegalodonModel::MegalodonModel(size_t input_dim, size_t d_model, size_t output_dim,
                               size_t num_layers, size_t num_heads, size_t ffn_mult,
                               size_t num_groups, size_t cema_ndim, size_t chunk_size)
    : input_dim_(input_dim),
      d_model_(d_model),
      output_dim_(output_dim),
      num_layers_(num_layers),
      num_heads_(num_heads),
      ffn_mult_(ffn_mult),
      num_groups_(num_groups),
      cema_ndim_(cema_ndim),
      chunk_size_(chunk_size),
      input_proj(input_dim, d_model),
      final_tsn(d_model, num_groups),
      output_proj(d_model, output_dim) {
    if (input_dim == 0)
        throw std::invalid_argument("MegalodonModel: input_dim must be > 0");
    if (d_model == 0)
        throw std::invalid_argument("MegalodonModel: d_model must be > 0");
    if (output_dim == 0)
        throw std::invalid_argument("MegalodonModel: output_dim must be > 0");
    if (num_layers == 0)
        throw std::invalid_argument("MegalodonModel: num_layers must be > 0");
    for (size_t i = 0; i < num_layers; ++i) {
        blocks.push_back(std::unique_ptr<MegalodonBlock>(
            new MegalodonBlock(d_model, num_heads, ffn_mult, num_groups, cema_ndim, chunk_size)));
    }
}

Tensor MegalodonModel::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("MegalodonModel::forward: input.cols != input_dim_");
    Tensor h = input_proj.forward(input);
    for (auto& b : blocks) h = b->forward(h);
    h = final_tsn.forward(h);
    return output_proj.forward(h);
}

Tensor MegalodonModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    Tensor d = output_proj.backward(grad_output, 0.0);
    d = final_tsn.backward(d, 0.0);
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        d = (*it)->backward(d, 0.0);
    }
    return input_proj.backward(d, 0.0);
}

void MegalodonModel::update_weights(double lr) {
    input_proj.update_weights(lr);
    final_tsn.update_weights(lr);
    output_proj.update_weights(lr);
    for (auto& b : blocks) b->update_weights(lr);
}

void MegalodonModel::zero_grad() {
    input_proj.zero_grad();
    final_tsn.zero_grad();
    output_proj.zero_grad();
    for (auto& b : blocks) b->zero_grad();
}

std::vector<Tensor*> MegalodonModel::parameters() {
    std::vector<Tensor*> p = input_proj.parameters();
    for (auto& b : blocks) {
        auto bp = b->parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    p.push_back(&final_tsn.gamma);
    p.push_back(&final_tsn.beta);
    auto op = output_proj.parameters();
    p.insert(p.end(), op.begin(), op.end());
    return p;
}

std::vector<Tensor*> MegalodonModel::gradients() {
    std::vector<Tensor*> g = input_proj.gradients();
    for (auto& b : blocks) {
        auto bg = b->gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    g.push_back(&final_tsn.grad_gamma_);
    g.push_back(&final_tsn.grad_beta_);
    auto og = output_proj.gradients();
    g.insert(g.end(), og.begin(), og.end());
    return g;
}
