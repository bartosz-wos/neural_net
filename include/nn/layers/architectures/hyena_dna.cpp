#include "hyena_dna.h"
#include <cmath>
#include <stdexcept>

// ============================================================================
// HyenaDNA — arXiv:2306.15794 §3.1/§3.2/§3.3. See hyena_dna.h for the
// operator derivation and the reference-implementation provenance.
// ============================================================================

// ---------------------------------------------------------------------------
// Helper: naive causal depthwise 1-D convolution over a (L, C) tensor.
//   y[t][c] = b[c] + sum_{j=0..S-1, t-j >= 0} W[c][j] * x[t-j][c]
// Causal left-padding only (pad_left = S-1, pad_right = 0), matching the
// reference `nn.Conv1d(..., padding=short_filter_order - 1)` followed by the
// `[..., :l]` truncation, which is a causal conv.
// ---------------------------------------------------------------------------
static Tensor causal_depthwise_conv1d(const Tensor& x, const Tensor& W,
                                      const Tensor& b) {
    size_t L = x.rows;
    size_t C = x.cols;
    size_t S = W.cols;
    Tensor y(L, C);
    for (size_t t = 0; t < L; ++t) {
        for (size_t c = 0; c < C; ++c) {
            double acc = b[c][0];
            for (size_t j = 0; j < S; ++j) {
                if (j > t) break;            // causal: no future taps
                acc += W[c][j] * x[t - j][c];
            }
            y[t][c] = acc;
        }
    }
    return y;
}

// ---------------------------------------------------------------------------
// Helper: causal long convolution of a single (L,) signal with a (L,) filter,
//   y[t] = sum_{s=0..t} k[s] * v[t-s]
// Naive O(L^2) — see the header for why (FFT is a drop-in upgrade).
// ---------------------------------------------------------------------------
static void causal_long_conv_1d(const std::vector<double>& v,
                                const std::vector<double>& k, size_t L,
                                std::vector<double>& y) {
    y.assign(L, 0.0);
    for (size_t t = 0; t < L; ++t) {
        double acc = 0.0;
        for (size_t s = 0; s <= t; ++s) acc += k[s] * v[t - s];
        y[t] = acc;
    }
}

// ===========================================================================
// HyenaDNAFilter
// ===========================================================================

HyenaDNAFilter::HyenaDNAFilter(size_t d_model, size_t l_max, size_t filter_order,
                               size_t emb_dim, size_t num_inner)
    : d_model_(d_model), l_max_(l_max), filter_order_(filter_order),
      emb_dim_(emb_dim), num_inner_(num_inner),
      mlp_in_W(Tensor::zeros(filter_order, emb_dim)),
      mlp_in_b(Tensor::zeros(1, filter_order)),
      sin_freq(Tensor::zeros(1, filter_order)),
      mlp_out_W(Tensor::zeros(d_model, filter_order)),
      deltas(Tensor::zeros(1, d_model)),
      bias(Tensor::zeros(1, d_model)) {
    // Validate before allocating: Tensor(0, N) is a legal but useless object,
    // and the shipped repo convention is to throw on a degenerate config.
    if (d_model == 0) throw std::invalid_argument("HyenaDNAFilter: d_model must be > 0");
    if (l_max == 0) throw std::invalid_argument("HyenaDNAFilter: l_max must be > 0");
    if (filter_order == 0)
        throw std::invalid_argument("HyenaDNAFilter: filter_order must be > 0");
    // The reference implementation asserts emb_dim is odd and >= 3 because the
    // positional embedding is [time, cos bands, sin bands].
    if (emb_dim < 3 || emb_dim % 2 == 0)
        throw std::invalid_argument("HyenaDNAFilter: emb_dim must be odd and >= 3");

    mlp_in_W = Tensor::random(filter_order, emb_dim, 0.3);
    for (size_t j = 0; j < filter_order; ++j) sin_freq[0][j] = 1.0;

    mlp_W.resize(num_inner);
    mlp_b.resize(num_inner);
    for (size_t i = 0; i < num_inner; ++i) {
        mlp_W[i] = Tensor::random(filter_order, filter_order, 0.3);
        mlp_b[i] = Tensor::zeros(1, filter_order);
    }
    // Final projection has NO bias (reference: nn.Linear(order, d_model, bias=False)).
    mlp_out_W = Tensor::random(d_model, filter_order, 0.3);

    // ExponentialModulation (reference l. 134-155):
    //   max_decay = log(target)/fast_pct,  min_decay = log(target)/slow_pct
    //   deltas    = linspace(min_decay, max_decay, d_model)
    // linspace semantics: the step is (max-min)/(n-1), so for d_model == 1
    // the result is the START value (min_decay), not the end. Verified
    // against numpy.linspace(mind, maxd, 1) == [mind].
    // Note the reference registers deltas with lr=0 (frozen); we make it a
    // normal learnable tensor and document the divergence in the header.
    {
        double log_target = std::log(kModulationTarget);
        double max_decay = log_target / kFastDecayPct;
        double min_decay = log_target / kSlowDecayPct;
        for (size_t c = 0; c < d_model; ++c) {
            deltas[0][c] = (d_model == 1)
                               ? min_decay
                               : min_decay + (double)c * (max_decay - min_decay) /
                                                  (double)(d_model - 1);
        }
    }

    grad_mlp_in_W = Tensor::zeros(filter_order, emb_dim);
    grad_mlp_in_b = Tensor::zeros(1, filter_order);
    grad_sin_freq = Tensor::zeros(1, filter_order);
    grad_mlp_W.resize(num_inner);
    grad_mlp_b.resize(num_inner);
    for (size_t i = 0; i < num_inner; ++i) {
        grad_mlp_W[i] = Tensor::zeros(filter_order, filter_order);
        grad_mlp_b[i] = Tensor::zeros(1, filter_order);
    }
    grad_mlp_out_W = Tensor::zeros(d_model, filter_order);
    grad_deltas = Tensor::zeros(1, d_model);
    grad_bias = Tensor::zeros(1, d_model);
}

Tensor HyenaDNAFilter::positional_embedding(size_t L) const {
    if (L > l_max_)
        throw std::invalid_argument("HyenaDNAFilter: L > l_max");
    Tensor z(L, emb_dim_);
    size_t bands = (emb_dim_ - 1) / 2;
    const double TWO_PI = 2.0 * 3.14159265358979323846;
    for (size_t l = 0; l < L; ++l) {
        // t = linspace(0, 1, L). For L == 1 that is [0.0], NOT [1.0] — this
        // matches torch.linspace(0, 1, 1) == [0.0], and it matters: setting
        // t=1 at L=1 would apply the FULL exp(-|deltas|) decay to the single
        // timestep and shrink the filter by ~20x versus the reference.
        z[l][0] = (L == 1) ? 0.0 : (double)l / (double)(L - 1);
        if (bands == 0) continue;
        // t_rescaled = linspace(0, L-1, L); w = 2*pi*t_rescaled/L
        double w = TWO_PI * (double)l / (double)L;
        for (size_t b = 0; b < bands; ++b) {
            // f = linspace(1e-4, bands-1, bands)
            double f;
            if (bands == 1) {
                f = 1e-4;
            } else {
                f = 1e-4 + (double)b * (double)(bands - 1 - 1e-4) / (double)(bands - 1);
            }
            // z = exp(-i*f*w) => [cos(f*w), -sin(f*w)]
            z[l][1 + b] = std::cos(f * w);
            z[l][1 + bands + b] = -std::sin(f * w);
        }
    }
    return z;
}

Tensor HyenaDNAFilter::filter(size_t L) {
    if (L > l_max_)
        throw std::invalid_argument("HyenaDNAFilter: L > l_max");

    // z = positional embedding, (L, emb_dim)
    Tensor z = positional_embedding(L);
    last_z = z;

    // --- layer 0: y0 = sin(freq * (W1 z + b1)),  y0 is (L, P)
    // Cache the Sin PRE-activation so the backward can apply
    // d/dx sin(f*x) = f*cos(f*x) with f the cached per-channel frequency.
    Tensor h1 = z * mlp_in_W.transpose();      // (L, emb_dim) @ (emb_dim, P)
    for (size_t p = 0; p < filter_order_; ++p)
        for (size_t l = 0; l < L; ++l) h1[l][p] += mlp_in_b[0][p];
    Tensor pre0 = h1.clone();
    for (size_t p = 0; p < filter_order_; ++p) {
        double f = sin_freq[0][p];
        for (size_t l = 0; l < L; ++l) h1[l][p] = std::sin(f * h1[l][p]);
    }
    last_pre.clear();
    last_post.clear();
    last_pre.push_back(pre0);
    last_post.push_back(h1);

    // --- inner layers i = 0..num_inner-1: y = sin(freq * (W y + b))
    Tensor cur = h1;
    for (size_t i = 0; i < num_inner_; ++i) {
        Tensor nxt = cur * mlp_W[i].transpose();     // (L, P)
        for (size_t p = 0; p < filter_order_; ++p)
            for (size_t l = 0; l < L; ++l) nxt[l][p] += mlp_b[i][0][p];
        Tensor pre = nxt.clone();
        for (size_t p = 0; p < filter_order_; ++p) {
            double f = sin_freq[0][p];
            for (size_t l = 0; l < L; ++l) nxt[l][p] = std::sin(f * nxt[l][p]);
        }
        last_pre.push_back(pre);
        last_post.push_back(nxt);
        cur = nxt;
    }

    // --- final projection, NO bias (reference: nn.Linear(P, D_f, bias=False))
    Tensor h0 = cur * mlp_out_W.transpose();        // (L, D_f)
    last_h0 = h0;

    // --- ExponentialModulation: h = (exp(-t*|deltas|) + shift) * h0
    // t is the normalized time channel of z, i.e. z[l][0].
    Tensor decay(L, d_model_);
    Tensor h(L, d_model_);
    for (size_t l = 0; l < L; ++l) {
        double t = z[l][0];
        for (size_t c = 0; c < d_model_; ++c) {
            double d = std::exp(-t * std::fabs(deltas[0][c])) + kModulationShift;
            decay[l][c] = d;
            h[l][c] = d * h0[l][c];
        }
    }
    last_decay = decay;
    return h;
}

void HyenaDNAFilter::backward(const Tensor& grad_h) {
    size_t L = grad_h.rows;
    if (grad_h.cols != d_model_)
        throw std::invalid_argument("HyenaDNAFilter: grad_h width mismatch");
    if (last_pre.empty())
        throw std::runtime_error("HyenaDNAFilter::backward: forward() must run first");
    size_t nlayers = num_inner_ + 1;   // 1 input Linear + num_inner inner Linears

    // ---- ExponentialModulation backward.
    // h[l][c] = decay[l][c] * h0[l][c],  decay = exp(-t_l * |deltas_c|) + shift
    // dL/dh0[l][c] = grad_h[l][c] * decay[l][c]
    // dL/ddeltas_c = sum_l grad_h[l][c] * h0[l][c] * d decay / d deltas_c
    //   d decay / d deltas_c = -t_l * sign(deltas_c) * exp(-t_l*|deltas_c|)
    Tensor gh0(L, d_model_);
    for (size_t c = 0; c < d_model_; ++c) {
        double sgn = (deltas[0][c] >= 0.0) ? 1.0 : -1.0;
        double acc = 0.0;
        for (size_t l = 0; l < L; ++l) {
            double t = last_z[l][0];
            gh0[l][c] = grad_h[l][c] * last_decay[l][c];
            // d/ddeltas [exp(-t*|d|)] = -t*sgn(d)*exp(-t*|d|)
            acc += grad_h[l][c] * last_h0[l][c] * (-t * sgn *
                                                  std::exp(-t * std::fabs(deltas[0][c])));
        }
        grad_deltas[0][c] += acc;
    }

    // ---- final projection backward: h0 = cur * mlp_out_W^T
    //   h0[l][c]   = sum_p cur[l][p] * mlp_out_W[c][p]
    //   dcur[l][p] += sum_c gh0[l][c] * mlp_out_W[c][p]
    //   dW[c][p]   += sum_l gh0[l][c] * cur[l][p]
    Tensor gcur = gh0 * mlp_out_W;          // (L, d_model) @ (d_model, P)
    for (size_t c = 0; c < d_model_; ++c)
        for (size_t p = 0; p < filter_order_; ++p) {
            double acc = 0.0;
            for (size_t l = 0; l < L; ++l) acc += gh0[l][c] * last_post[nlayers - 1][l][p];
            grad_mlp_out_W[c][p] += acc;
        }

    // ---- walk the Sin layers back to front.
    // Layer i (0-based, 0 = input Linear): pre = prev * W^T + b; post = sin(f*pre)
    //   d pre      = g_post * f * cos(f * pre)         (f = sin_freq[p], cached)
    //   d W[p][k] += sum_l d_pre[l][p] * prev[l][k]
    //   d b[p]    += sum_l d_pre[l][p]
    for (size_t li = nlayers; li-- > 0;) {
        const Tensor& pre = last_pre[li];
        const Tensor& post = last_post[li];
        Tensor dpre(L, filter_order_);
        for (size_t p = 0; p < filter_order_; ++p) {
            double f = sin_freq[0][p];
            for (size_t l = 0; l < L; ++l)
                dpre[l][p] = gcur[l][p] * f * std::cos(f * pre[l][p]);
        }
        // d sin_freq[p] = sum_l g_post[l][p] * pre[l][p] * cos(f*pre[l][p])
        for (size_t p = 0; p < filter_order_; ++p) {
            double f = sin_freq[0][p];
            double acc = 0.0;
            for (size_t l = 0; l < L; ++l)
                acc += gcur[l][p] * pre[l][p] * std::cos(f * pre[l][p]);
            grad_sin_freq[0][p] += acc;
        }

        const Tensor& prev = (li == 0) ? last_z : last_post[li - 1];
        size_t prev_cols = (li == 0) ? emb_dim_ : filter_order_;
        Tensor* W = (li == 0) ? &mlp_in_W : &mlp_W[li - 1];
        Tensor* gW = (li == 0) ? &grad_mlp_in_W : &grad_mlp_W[li - 1];
        Tensor* gb = (li == 0) ? &grad_mlp_in_b : &grad_mlp_b[li - 1];
        for (size_t p = 0; p < filter_order_; ++p) {
            for (size_t k = 0; k < prev_cols; ++k) {
                double acc = 0.0;
                for (size_t l = 0; l < L; ++l) acc += dpre[l][p] * prev[l][k];
                (*gW)[p][k] += acc;
            }
            double accb = 0.0;
            for (size_t l = 0; l < L; ++l) accb += dpre[l][p];
            (*gb)[0][p] += accb;
        }
        (void)post;
        if (li > 0) {
            // d prev = dpre * W   (L,P) @ (P,P)
            Tensor gprev = dpre * (*W);      // (L, P)
            gcur = gprev;
        }
        // For li == 0 the gradient w.r.t. z is not needed (z is not learnable).
    }
}

void HyenaDNAFilter::zero_grad() {
    grad_mlp_in_W.fill(0.0);
    grad_mlp_in_b.fill(0.0);
    grad_sin_freq.fill(0.0);
    for (size_t i = 0; i < num_inner_; ++i) {
        grad_mlp_W[i].fill(0.0);
        grad_mlp_b[i].fill(0.0);
    }
    grad_mlp_out_W.fill(0.0);
    grad_deltas.fill(0.0);
    grad_bias.fill(0.0);
}

void HyenaDNAFilter::update_weights(double lr) {
    // parameters() and gradients() are index-aligned by construction (see the
    // two methods below), so a positional zip is the whole step.
    std::vector<Tensor*> ps = parameters();
    std::vector<Tensor*> gs = gradients();
    for (size_t i = 0; i < ps.size(); ++i)
        for (size_t r = 0; r < ps[i]->rows; ++r)
            for (size_t c = 0; c < ps[i]->cols; ++c)
                (*ps[i])[r][c] -= lr * (*gs[i])[r][c];
}

std::vector<Tensor*> HyenaDNAFilter::parameters() {
    std::vector<Tensor*> p = {&mlp_in_W, &mlp_in_b, &sin_freq};
    for (size_t i = 0; i < num_inner_; ++i) {
        p.push_back(&mlp_W[i]);
        p.push_back(&mlp_b[i]);
    }
    p.push_back(&mlp_out_W);
    p.push_back(&deltas);
    p.push_back(&bias);
    return p;
}

std::vector<Tensor*> HyenaDNAFilter::gradients() {
    std::vector<Tensor*> g = {&grad_mlp_in_W, &grad_mlp_in_b, &grad_sin_freq};
    for (size_t i = 0; i < num_inner_; ++i) {
        g.push_back(&grad_mlp_W[i]);
        g.push_back(&grad_mlp_b[i]);
    }
    g.push_back(&grad_mlp_out_W);
    g.push_back(&grad_deltas);
    g.push_back(&grad_bias);
    return g;
}

// ===========================================================================
// HyenaDNAOperator
// ===========================================================================

HyenaDNAOperator::HyenaDNAOperator(size_t d_model, size_t l_max, size_t num_heads,
                                   size_t order, size_t filter_order,
                                   size_t short_filter_order)
    : d_model_(d_model), l_max_(l_max), num_heads_(num_heads), order_(order),
      filter_order_(filter_order), short_filter_order_(short_filter_order),
      // The `? : 1` guards keep every Tensor/Dense constructor well-formed
      // (no zero-size allocation) AND keep the filter's width expression
      // division-free, so the validation in the body can still throw before
      // any real allocation. Same placeholder-then-validate idiom as MinLSTM.
      in_proj(d_model ? d_model : 1,
              (order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1),
      out_proj(d_model ? d_model : 1, d_model ? d_model : 1),
      short_W(Tensor::zeros((order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1,
                            short_filter_order ? short_filter_order : 1)),
      short_b(Tensor::zeros((order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1, 1)),
      grad_short_W(Tensor::zeros((order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1,
                                 short_filter_order ? short_filter_order : 1)),
      grad_short_b(Tensor::zeros((order >= 2) ? (order + 1) * (d_model ? d_model : 1) : 1, 1)),
      filter(num_heads && d_model && order >= 2
                 ? (d_model / num_heads) * (order - 1)
                 : 1,
             l_max ? l_max : 1, filter_order ? filter_order : 1) {
    if (d_model == 0)
        throw std::invalid_argument("HyenaDNAOperator: d_model must be > 0");
    if (l_max == 0)
        throw std::invalid_argument("HyenaDNAOperator: l_max must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("HyenaDNAOperator: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument("HyenaDNAOperator: d_model must be divisible by num_heads");
    // The reference implementation asserts order >= 2.
    if (order < 2)
        throw std::invalid_argument("HyenaDNAOperator: order must be >= 2");
    if (short_filter_order == 0)
        throw std::invalid_argument("HyenaDNAOperator: short_filter_order must be > 0");
    if (filter_order == 0)
        throw std::invalid_argument("HyenaDNAOperator: filter_order must be > 0");

    head_dim_ = d_model / num_heads;

    short_W = Tensor::random((order + 1) * d_model, short_filter_order, 0.2);
}

Tensor HyenaDNAOperator::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("HyenaDNAOperator: input width must equal d_model");
    if (input.rows > l_max_)
        throw std::invalid_argument("HyenaDNAOperator: L > l_max");
    if (input.rows == 0)
        throw std::invalid_argument("HyenaDNAOperator: empty input");

    size_t L = input.rows;
    size_t H = num_heads_, HD = head_dim_, O = order_;
    last_input = input;

    // --- in_proj, applied per token. Dense is (batch, in) -> (batch, out) and a
    // (L, D) tensor is exactly a batch of L tokens, so it applies directly.
    Tensor u = in_proj.forward(input);            // (L, (O+1)*D)
    last_in_proj = u;

    // --- causal depthwise short convolution over the concatenated projections
    Tensor uc = causal_depthwise_conv1d(u, short_W, short_b);
    last_uc = uc;

    // --- shared filter bank: (L, head_dim*(O-1))
    Tensor k = filter.filter(L);
    last_filter_h = k;

    // Per-head scratch, all (L, D) so the backward can read them back directly.
    // The (L, (O+1)*D) projection tensor is indexed
    //   [l][h*(O+1)*HD + o*HD + c]
    // so head h owns gates o = 0..O-1 and the value v at o = O.
    last_gate.assign(O, Tensor(L, d_model_));
    last_v_before.assign(O, Tensor(L, d_model_));
    last_v_gated.assign(O, Tensor(L, d_model_));

    Tensor y(L, d_model_);
    std::vector<double> v_sig(L), k_sig(L), conv(L);

    for (size_t h = 0; h < H; ++h) {
        // Scatter this head's gates and value out of uc into the caches.
        for (size_t o = 0; o < O; ++o)
            for (size_t l = 0; l < L; ++l)
                for (size_t c = 0; c < HD; ++c)
                    last_gate[o][l][h * HD + c] =
                        uc[l][h * (O + 1) * HD + o * HD + c];

        // v is PER-CHANNEL: v[l][c] = uc[l][h*(O+1)*HD + O*HD + c].
        // Keeping v as (L, HD) is essential — a single scalar per timestep
        // would collapse the head's channels and silently compute a rank-1
        // operator instead of HD independent ones.
        std::vector<std::vector<double>> v(HD, std::vector<double>(L, 0.0));
        for (size_t c = 0; c < HD; ++c)
            for (size_t l = 0; l < L; ++l)
                v[c][l] = uc[l][h * (O + 1) * HD + O * HD + c];

        // --- the recurrence, for o = O-1 down to 1 (matching the reference's
        //     `for o, x_i in enumerate(reversed(x[1:]))`):
        //         v = v * x[o]                          (element-wise gate D_x)
        //         v = long_conv(v, k_o) + bias_o * v    (T_h then the skip)
        // Filter k_o for head h is k[:, h*(O-1) + o] and the skip is
        // bias[:, h*(O-1) + o] — the reference index layout
        // `rearrange(k, "c l (v o) -> c o v l", v=head_dim, o=order-1)`.
        for (size_t o = O; o-- > 1;) {
            for (size_t c = 0; c < HD; ++c) {
                for (size_t l = 0; l < L; ++l) {
                    double before = v[c][l];
                    double g = last_gate[o][l][h * HD + c];
                    last_v_before[o][l][h * HD + c] = before;
                    last_v_gated[o][l][h * HD + c] = before * g;
                    v_sig[l] = before * g;
                }
                for (size_t l = 0; l < L; ++l) k_sig[l] = k[l][h * (O - 1) + o];
                causal_long_conv_1d(v_sig, k_sig, L, conv);
                double b = filter.bias[0][h * (O - 1) + o];
                for (size_t l = 0; l < L; ++l) {
                    v_sig[l] = conv[l] + b * v_sig[l];
                    v[c][l] = v_sig[l];
                }
            }
        }

        // --- final gate by x[0], scattered into the head's output columns.
        for (size_t l = 0; l < L; ++l) {
            for (size_t c = 0; c < HD; ++c) {
                last_v_before[0][l][h * HD + c] = v[c][l];
                y[l][h * HD + c] = v[c][l] * last_gate[0][l][h * HD + c];
            }
        }
    }

    last_y_pre_out = y;
    return out_proj.forward(y);
}

Tensor HyenaDNAOperator::backward(const Tensor&, double) {
    throw std::runtime_error("HyenaDNAOperator::backward: not implemented");
}

void HyenaDNAOperator::update_weights(double lr) {
    in_proj.update_weights(lr);
    out_proj.update_weights(lr);
    for (size_t i = 0; i < short_W.rows; ++i) {
        for (size_t j = 0; j < short_W.cols; ++j)
            short_W[i][j] -= lr * grad_short_W[i][j];
        short_b[i][0] -= lr * grad_short_b[i][0];
    }
    // The filter holds raw tensors, not Dense layers, so its SGD step is its
    // own method rather than a Dense::update_weights call.
    filter.update_weights(lr);
}

Tensor HyenaDNAOperator::get_weights() const { return in_proj.weights; }
Tensor HyenaDNAOperator::get_gradients() const { return in_proj.grad_weights; }

std::vector<Tensor*> HyenaDNAOperator::parameters() {
    std::vector<Tensor*> p = in_proj.parameters();
    for (auto* q : filter.parameters()) p.push_back(q);
    p.push_back(&out_proj.weights);
    p.push_back(&out_proj.bias);
    p.push_back(&short_W);
    p.push_back(&short_b);
    return p;
}

std::vector<Tensor*> HyenaDNAOperator::gradients() {
    std::vector<Tensor*> g = in_proj.gradients();
    for (auto* q : filter.gradients()) g.push_back(q);
    g.push_back(&out_proj.grad_weights);
    g.push_back(&out_proj.grad_bias);
    g.push_back(&grad_short_W);
    g.push_back(&grad_short_b);
    return g;
}

void HyenaDNAOperator::zero_grad() {
    in_proj.zero_grad();
    out_proj.zero_grad();
    filter.zero_grad();
    grad_short_W.fill(0.0);
    grad_short_b.fill(0.0);
}

// ===========================================================================
// HyenaDNABlock
// ===========================================================================

HyenaDNABlock::HyenaDNABlock(size_t d_model, size_t l_max, size_t num_heads,
                             size_t order, size_t filter_order, size_t ffn_mult)
    : d_model_(d_model), l_max_(l_max), ffn_mult_(ffn_mult),
      ln1(d_model ? d_model : 1), ln2(d_model ? d_model : 1),
      op(d_model, l_max, num_heads, order, filter_order),
      ffn1(d_model ? d_model : 1,
           (ffn_mult > 0) ? ffn_mult * (d_model ? d_model : 1) : 1),
      ffn2((ffn_mult > 0) ? ffn_mult * (d_model ? d_model : 1) : 1,
           d_model ? d_model : 1) {
    if (d_model == 0)
        throw std::invalid_argument("HyenaDNABlock: d_model must be > 0");
    if (l_max == 0)
        throw std::invalid_argument("HyenaDNABlock: l_max must be > 0");
}

Tensor HyenaDNABlock::forward(const Tensor&) {
    throw std::runtime_error("HyenaDNABlock::forward: not implemented");
}

Tensor HyenaDNABlock::backward(const Tensor&, double) {
    throw std::runtime_error("HyenaDNABlock::backward: not implemented");
}

void HyenaDNABlock::update_weights(double lr) {
    op.update_weights(lr);
    if (ffn_mult_ > 0) {
        ffn1.update_weights(lr);
        ffn2.update_weights(lr);
    }
    ln1.update_weights(lr);
    ln2.update_weights(lr);
}

Tensor HyenaDNABlock::get_weights() const { return op.in_proj.weights; }
Tensor HyenaDNABlock::get_gradients() const { return op.in_proj.grad_weights; }

std::vector<Tensor*> HyenaDNABlock::parameters() {
    std::vector<Tensor*> p = ln1.parameters();
    for (auto* q : op.parameters()) p.push_back(q);
    if (ffn_mult_ > 0) {
        for (auto* q : ffn1.parameters()) p.push_back(q);
        for (auto* q : ffn2.parameters()) p.push_back(q);
    }
    for (auto* q : ln2.parameters()) p.push_back(q);
    return p;
}

std::vector<Tensor*> HyenaDNABlock::gradients() {
    std::vector<Tensor*> g = ln1.gradients();
    for (auto* q : op.gradients()) g.push_back(q);
    if (ffn_mult_ > 0) {
        for (auto* q : ffn1.gradients()) g.push_back(q);
        for (auto* q : ffn2.gradients()) g.push_back(q);
    }
    for (auto* q : ln2.gradients()) g.push_back(q);
    return g;
}

void HyenaDNABlock::zero_grad() {
    ln1.zero_grad();
    op.zero_grad();
    if (ffn_mult_ > 0) {
        ffn1.zero_grad();
        ffn2.zero_grad();
    }
    ln2.zero_grad();
}

// ===========================================================================
// HyenaDNAModel
// ===========================================================================

HyenaDNAModel::HyenaDNAModel(size_t d_model, size_t l_max, size_t depth,
                             size_t num_classes, size_t num_heads, size_t order,
                             size_t filter_order, size_t ffn_mult)
    : d_model_(d_model), l_max_(l_max), depth_(depth), num_classes_(num_classes),
      classifier(d_model ? d_model : 1, num_classes ? num_classes : 1) {
    if (d_model == 0)
        throw std::invalid_argument("HyenaDNAModel: d_model must be > 0");
    if (l_max == 0)
        throw std::invalid_argument("HyenaDNAModel: l_max must be > 0");
    if (depth == 0)
        throw std::invalid_argument("HyenaDNAModel: depth must be > 0");
    if (num_classes == 0)
        throw std::invalid_argument("HyenaDNAModel: num_classes must be > 0");

    blocks.reserve(depth);
    for (size_t i = 0; i < depth; ++i) {
        blocks.emplace_back(d_model, l_max, num_heads, order, filter_order,
                           ffn_mult);
    }
}

Tensor HyenaDNAModel::forward(const Tensor&) {
    throw std::runtime_error("HyenaDNAModel::forward: not implemented");
}

Tensor HyenaDNAModel::backward(const Tensor&, double) {
    throw std::runtime_error("HyenaDNAModel::backward: not implemented");
}

void HyenaDNAModel::update_weights(double lr) {
    for (size_t i = 0; i < depth_; ++i) blocks[i].update_weights(lr);
    classifier.update_weights(lr);
}

Tensor HyenaDNAModel::get_weights() const {
    return blocks.empty() ? Tensor(0, 0) : blocks[0].op.in_proj.weights;
}
Tensor HyenaDNAModel::get_gradients() const {
    return blocks.empty() ? Tensor(0, 0) : blocks[0].op.in_proj.grad_weights;
}

std::vector<Tensor*> HyenaDNAModel::parameters() {
    std::vector<Tensor*> p;
    for (size_t i = 0; i < depth_; ++i)
        for (auto* q : blocks[i].parameters()) p.push_back(q);
    for (auto* q : classifier.parameters()) p.push_back(q);
    return p;
}

std::vector<Tensor*> HyenaDNAModel::gradients() {
    std::vector<Tensor*> g;
    for (size_t i = 0; i < depth_; ++i)
        for (auto* q : blocks[i].gradients()) g.push_back(q);
    for (auto* q : classifier.gradients()) g.push_back(q);
    return g;
}

void HyenaDNAModel::zero_grad() {
    for (size_t i = 0; i < depth_; ++i) blocks[i].zero_grad();
    classifier.zero_grad();
}

// ===========================================================================
// SequenceLengthWarmup (arXiv:2306.15794 §3.2)
// ===========================================================================

SequenceLengthWarmup::SequenceLengthWarmup(size_t start_len, size_t num_stages,
                                           size_t epoch_scale)
    : start_len_(start_len), num_stages_(num_stages), epoch_scale_(epoch_scale) {
    if (start_len == 0)
        throw std::invalid_argument("SequenceLengthWarmup: start_len must be > 0");
    if (num_stages == 0)
        throw std::invalid_argument("SequenceLengthWarmup: num_stages must be > 0");
    if (epoch_scale == 0)
        throw std::invalid_argument("SequenceLengthWarmup: epoch_scale must be > 0");
}

size_t SequenceLengthWarmup::stage_for_epoch(size_t epoch) const {
    size_t stage = 0;
    size_t stage_start = 0;
    for (size_t s = 0; s < num_stages_; ++s) {
        size_t stage_len = epoch_scale_ << s;   // epoch_scale * 2^s
        if (epoch < stage_start + stage_len) return s;
        stage_start += stage_len;
        ++stage;
    }
    return num_stages_ - 1;   // saturate at the final stage
}

size_t SequenceLengthWarmup::seq_len_for_epoch(size_t epoch) const {
    return start_len_ << stage_for_epoch(epoch);
}

size_t SequenceLengthWarmup::total_epochs() const {
    return epoch_scale_ * ((size_t(1) << num_stages_) - 1);
}

size_t SequenceLengthWarmup::max_seq_len() const {
    return start_len_ << (num_stages_ - 1);
}

// ===========================================================================
// SoftPrompting (arXiv:2306.15794 §3.3, Eq. 3.2)
// ===========================================================================

SoftPrompting::SoftPrompting(size_t prompt_len, size_t d_model, bool at_front)
    : prompt_len_(prompt_len), d_model_(d_model), at_front_(at_front) {
    if (prompt_len == 0)
        throw std::invalid_argument("SoftPrompting: prompt_len must be > 0");
    if (d_model == 0)
        throw std::invalid_argument("SoftPrompting: d_model must be > 0");
    prompt_ = Tensor::random(prompt_len, d_model, 0.02);
    grad_prompt_ = Tensor::zeros(prompt_len, d_model);
}

Tensor SoftPrompting::forward(const Tensor& embedded) {
    if (embedded.cols != d_model_)
        throw std::invalid_argument("SoftPrompting: embedded d_model mismatch");
    last_embedded = embedded;
    size_t T = embedded.rows;
    Tensor out(T + prompt_len_, d_model_);
    for (size_t t = 0; t < T; ++t)
        for (size_t c = 0; c < d_model_; ++c)
            out[at_front_ ? (t + prompt_len_) : t][c] = embedded[t][c];
    for (size_t n = 0; n < prompt_len_; ++n)
        for (size_t c = 0; c < d_model_; ++c)
            out[at_front_ ? n : (T + n)][c] = prompt_[n][c];
    return out;
}

Tensor SoftPrompting::backward(const Tensor& grad_output, double) {
    if (grad_output.rows != last_embedded.rows + prompt_len_)
        throw std::invalid_argument("SoftPrompting: grad_output shape mismatch");
    for (size_t n = 0; n < prompt_len_; ++n)
        for (size_t c = 0; c < d_model_; ++c)
            grad_prompt_[n][c] += grad_output[at_front_ ? n : (last_embedded.rows + n)][c];
    // The embedded input is fixed (a frozen embedding table in the paper's
    // setting), so the gradient w.r.t. it is zero.
    return Tensor::zeros(last_embedded.rows, last_embedded.cols);
}

void SoftPrompting::update_weights(double lr) {
    for (size_t n = 0; n < prompt_len_; ++n)
        for (size_t c = 0; c < d_model_; ++c)
            prompt_[n][c] -= lr * grad_prompt_[n][c];
}

void SoftPrompting::zero_grad() { grad_prompt_.fill(0.0); }
