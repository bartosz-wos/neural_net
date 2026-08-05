#include "hyena.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <cstdint>

// ============================================================================
// Forward declarations for static cache utilities.
// ============================================================================
static std::atomic<uint64_t>& cache_gen_counter();
struct HyenaFilterCache;
static std::mt19937& hyena_gen() {
    static std::mt19937 gen(42);
    return gen;
}
static void fill_normal(Tensor& t, double stddev) {
    std::normal_distribution<> dis(0.0, stddev);
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = dis(hyena_gen());
}
[[maybe_unused]] static void fill_uniform(Tensor& t, double lo, double hi) {
    (void)t; (void)lo; (void)hi;
    // Reserved for future use (uniform init schemes); kept to silence the
    // unused-static warning when other utilities are added.
}

// ============================================================================
// Helper: 1D causal convolution.
//   y[t] = sum_{s=0..t} k[s] * v[t-s]
// Operates on (L,) tensors (we treat one channel at a time, batched externally).
// ============================================================================
static Tensor causal_conv_1d(const Tensor& v, const Tensor& k) {
    size_t L = v.cols;  // 1 x L
    Tensor y(1, L);
    y.fill(0.0);
    for (size_t t = 0; t < L; ++t) {
        double acc = 0.0;
        for (size_t s = 0; s <= t; ++s) {
            acc += k[0][s] * v[0][t - s];
        }
        y[0][t] = acc;
    }
    return y;
}

// dL/dv[t'] = sum_t grad_y[t] * k[t - t'], where the inner sum has t >= t'
static Tensor causal_conv_1d_backward_v(const Tensor& grad_y, const Tensor& k) {
    size_t L = grad_y.cols;
    Tensor grad_v(1, L);
    grad_v.fill(0.0);
    for (size_t t_prime = 0; t_prime < L; ++t_prime) {
        double acc = 0.0;
        // t ranges from t_prime to L-1, k index = t - t_prime (>= 0, <= L-1-t_prime <= L-1)
        for (size_t t = t_prime; t < L; ++t) {
            acc += grad_y[0][t] * k[0][t - t_prime];
        }
        grad_v[0][t_prime] = acc;
    }
    return grad_v;
}

// dL/dk[s] = sum_t grad_y[t] * v[t - s], where t >= s
static Tensor causal_conv_1d_backward_k(const Tensor& grad_y, const Tensor& v) {
    size_t L = grad_y.cols;
    Tensor grad_k(1, L);
    grad_k.fill(0.0);
    for (size_t s = 0; s < L; ++s) {
        double acc = 0.0;
        for (size_t t = s; t < L; ++t) {
            acc += grad_y[0][t] * v[0][t - s];
        }
        grad_k[0][s] = acc;
    }
    return grad_k;
}

// ============================================================================
// Helper: naive Linear forward (y = x @ W^T + b).
// x: (rows, in), W: (out, in), b: (1, out) → y: (rows, out).
// ============================================================================
static Tensor linear_forward(const Tensor& x, const Tensor& W, const Tensor& b) {
    size_t rows = x.rows;
    size_t in_f = W.cols;
    size_t out_f = W.rows;
    if (x.cols != in_f) throw std::invalid_argument("linear_forward: x.cols != W.cols");
    Tensor y(rows, out_f);
    y.fill(0.0);
    // y[i, j] = sum_k x[i, k] * W[j, k] + b[j]
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < out_f; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < in_f; ++k) acc += x[i][k] * W[j][k];
            y[i][j] = acc + b[0][j];
        }
    }
    return y;
}

// dgrad_x = grad_y @ W, dgrad_W = grad_y^T @ x, dgrad_b = sum_i grad_y[i]
static void linear_backward(const Tensor& x, const Tensor& W,
                            const Tensor& grad_y,
                            Tensor& grad_x, Tensor& grad_W, Tensor& grad_b) {
    size_t rows = x.rows;
    size_t in_f = W.cols;
    size_t out_f = W.rows;
    grad_x = Tensor(rows, in_f);
    grad_W = Tensor(out_f, in_f);
    grad_b = Tensor(1, out_f);
    grad_x.fill(0.0); grad_W.fill(0.0); grad_b.fill(0.0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < out_f; ++j) {
            double gy = grad_y[i][j];
            // grad_x[i, k] += grad_y[i, j] * W[j, k]
            for (size_t k = 0; k < in_f; ++k) grad_x[i][k] += gy * W[j][k];
            // grad_W[j, k] += grad_y[i, j] * x[i, k]
            for (size_t k = 0; k < in_f; ++k) grad_W[j][k] += gy * x[i][k];
            // grad_b[j] += grad_y[i, j]
            grad_b[0][j] += gy;
        }
    }
}

// Apply y = sin(freq * x) element-wise. (Backward is inlined inside
// HyenaFilter::backward for clarity of the gradient chain.)
static Tensor sin_forward(const Tensor& x, const Tensor& freq) {
    Tensor y(x.rows, x.cols);
    for (size_t i = 0; i < x.data.size(); ++i) {
        double w = freq.data[i % freq.data.size()];
        y.data[i] = std::sin(w * x.data[i]);
    }
    return y;
}

// ============================================================================
// HyenaFilter
// ============================================================================
HyenaFilter::HyenaFilter(size_t d_model, size_t l_max, size_t filter_order,
                         size_t emb_dim, size_t num_inner)
    : d_model(d_model), l_max(l_max), filter_order(filter_order),
      emb_dim(emb_dim), num_inner(num_inner),
      instance_gen(cache_gen_counter().fetch_add(1))
{
    if (emb_dim < 3 || (emb_dim % 2) == 0) {
        throw std::invalid_argument("HyenaFilter: emb_dim must be odd and >= 3");
    }
    // mlp_in_W (P, emb_dim), mlp_in_b (1, P)
    mlp_in_W = Tensor(filter_order, emb_dim);
    mlp_in_b = Tensor(1, filter_order);
    fill_normal(mlp_in_W, std::sqrt(2.0 / (emb_dim + filter_order)));
    fill_normal(mlp_in_b, 0.01);
    grad_mlp_in_W = Tensor(filter_order, emb_dim);
    grad_mlp_in_b = Tensor(1, filter_order);

    // sin_freq (1, P): init to a small positive value so sin(w*x) varies meaningfully
    sin_freq = Tensor(1, filter_order);
    for (size_t p = 0; p < filter_order; ++p) sin_freq[0][p] = 1.0;
    grad_sin_freq = Tensor(1, filter_order);

    // Inner MLP weights
    mlp_W.resize(num_inner);
    mlp_b.resize(num_inner);
    grad_mlp_W.resize(num_inner);
    grad_mlp_b.resize(num_inner);
    for (size_t k = 0; k < num_inner; ++k) {
        mlp_W[k] = Tensor(filter_order, filter_order);
        mlp_b[k] = Tensor(1, filter_order);
        fill_normal(mlp_W[k], std::sqrt(2.0 / (filter_order + filter_order)));
        fill_normal(mlp_b[k], 0.01);
        grad_mlp_W[k] = Tensor(filter_order, filter_order);
        grad_mlp_b[k] = Tensor(1, filter_order);
    }

    // mlp_out_W (D, P): final projection
    mlp_out_W = Tensor(d_model, filter_order);
    fill_normal(mlp_out_W, std::sqrt(2.0 / (filter_order + d_model)));
    grad_mlp_out_W = Tensor(d_model, filter_order);

    // deltas (1, D): per-channel decay rate, init small positive
    deltas = Tensor(1, d_model);
    for (size_t d = 0; d < d_model; ++d) deltas[0][d] = 0.5;
    grad_deltas = Tensor(1, d_model);

    // D_skip (1, D): skip parameter
    D_skip = Tensor(1, d_model);
    fill_normal(D_skip, 0.1);
    grad_D_skip = Tensor(1, d_model);
}

void HyenaFilter::zero_grad() {
    grad_mlp_in_W.fill(0.0);
    grad_mlp_in_b.fill(0.0);
    grad_sin_freq.fill(0.0);
    for (size_t k = 0; k < num_inner; ++k) {
        grad_mlp_W[k].fill(0.0);
        grad_mlp_b[k].fill(0.0);
    }
    grad_mlp_out_W.fill(0.0);
    grad_deltas.fill(0.0);
    grad_D_skip.fill(0.0);
}

std::vector<Tensor*> HyenaFilter::parameters() {
    std::vector<Tensor*> p = {&mlp_in_W, &mlp_in_b, &sin_freq, &mlp_out_W,
                              &deltas, &D_skip};
    for (auto& w : mlp_W) p.push_back(&w);
    for (auto& b : mlp_b) p.push_back(&b);
    return p;
}
std::vector<Tensor*> HyenaFilter::gradients() {
    std::vector<Tensor*> g = {&grad_mlp_in_W, &grad_mlp_in_b, &grad_sin_freq,
                              &grad_mlp_out_W, &grad_deltas, &grad_D_skip};
    for (auto& w : grad_mlp_W) g.push_back(&w);
    for (auto& b : grad_mlp_b) g.push_back(&b);
    return g;
}

// Generate positional embedding z of shape (L, emb_dim):
//   z[l, 0] = l / (L-1)                         (normalized time)
//   z[l, 2k+1] = cos(2π·k·l/L)
//   z[l, 2k+2] = sin(2π·k·l/L)
// for k = 0..(emb_dim-3)/2.
static Tensor positional_embedding(size_t L, size_t emb_dim) {
    Tensor z(L, emb_dim);
    for (size_t l = 0; l < L; ++l) {
        double t = (L > 1) ? static_cast<double>(l) / static_cast<double>(L - 1) : 0.0;
        z[l][0] = t;
        size_t bands = (emb_dim - 1) / 2;
        for (size_t k = 0; k < bands; ++k) {
            double angle = 2.0 * M_PI * static_cast<double>(k * l) / static_cast<double>(L);
            z[l][2 * k + 1] = std::cos(angle);
            z[l][2 * k + 2] = std::sin(angle);
        }
    }
    return z;
}

// Cache of filter activations for backward. Stored as a static thread_local
// for simplicity (single-threaded use), with a generation counter so that
// different HyenaFilter instances don't collide. Each HyenaFilter instance
// has a unique generation set in its constructor.
static std::atomic<uint64_t>& cache_gen_counter() {
    static std::atomic<uint64_t> c{1};
    return c;
}
struct HyenaFilterCache {
    uint64_t gen = 0;   // matches the instance that last populated this cache
    Tensor z;            // (L, emb_dim) — positional embedding (constant for given L)
    Tensor a0;           // (L, P) — pre-activation before first sin
    Tensor h0;           // (L, P) — post first sin
    std::vector<Tensor> a_inner;  // pre-activations inside inner MLPs
    std::vector<Tensor> h_inner;  // post-sin inside inner MLPs
    Tensor a_final;      // (L, P) — pre-activation before final projection
    Tensor pre_decay;    // (L, D) — h before modulation
    Tensor mod_t;        // (L, 1) — the time vector broadcasted (for grad_deltas)
    size_t L;
    size_t D;
    size_t P;
};
static HyenaFilterCache& cache() {
    static thread_local HyenaFilterCache c;
    return c;
}

std::pair<Tensor, Tensor> HyenaFilter::filter(size_t L) {
    if (L == 0) throw std::invalid_argument("HyenaFilter: L must be > 0");
    if (L > l_max) throw std::invalid_argument("HyenaFilter: L > l_max");

    HyenaFilterCache& c = cache();
    c.gen = instance_gen;  // mark this cache as belonging to this instance
    c.L = L; c.D = d_model; c.P = filter_order;
    c.a_inner.resize(num_inner);
    c.h_inner.resize(num_inner);

    // Positional embedding
    c.z = positional_embedding(L, emb_dim);

    // Linear in: (L, emb_dim) → (L, P)
    Tensor h = linear_forward(c.z, mlp_in_W, mlp_in_b);

    // First sin: y = sin(freq * h). pre_act = freq * h.
    c.a0 = Tensor(L, filter_order);
    for (size_t i = 0; i < h.data.size(); ++i) {
        double w = sin_freq.data[i % filter_order];
        c.a0.data[i] = w * h.data[i];
    }
    c.h0 = sin_forward(h, sin_freq);
    h = c.h0;

    // Inner MLPs
    for (size_t k = 0; k < num_inner; ++k) {
        Tensor z = linear_forward(h, mlp_W[k], mlp_b[k]);
        c.a_inner[k] = Tensor(L, filter_order);
        for (size_t i = 0; i < z.data.size(); ++i) {
            double w = sin_freq.data[i % filter_order];
            c.a_inner[k].data[i] = w * z.data[i];
        }
        c.h_inner[k] = sin_forward(z, sin_freq);
        h = c.h_inner[k];
    }

    // Final projection: (L, P) → (L, D). No bias.
    Tensor a = h;  // pre-activation for the final Linear
    c.a_final = a;
    // y[i, d] = sum_p mlp_out_W[d, p] * a[i, p]
    Tensor y(L, d_model);
    y.fill(0.0);
    for (size_t l = 0; l < L; ++l) {
        for (size_t d = 0; d < d_model; ++d) {
            double acc = 0.0;
            for (size_t p = 0; p < filter_order; ++p) acc += mlp_out_W[d][p] * a[l][p];
            y[l][d] = acc;
        }
    }

    // Exponential modulation: y[l, d] *= exp(-delta[d] * t[l]) where t[l] = l/(L-1)
    c.pre_decay = y;  // before modulation
    c.mod_t = Tensor(L, 1);
    for (size_t l = 0; l < L; ++l) {
        double t = (L > 1) ? static_cast<double>(l) / static_cast<double>(L - 1) : 0.0;
        c.mod_t[l][0] = t;
        for (size_t d = 0; d < d_model; ++d) {
            y[l][d] *= std::exp(-deltas[0][d] * t);
        }
    }

    return std::make_pair(y, D_skip);
}

void HyenaFilter::backward(const Tensor& grad_h, const Tensor& grad_D_skip_in) {
    // Accumulate D_skip gradient
    for (size_t d = 0; d < d_model; ++d) grad_D_skip[0][d] += grad_D_skip_in[0][d];

    // Derive L from grad_h (the upstream gradient shape). The cache may be
    // stale: (a) belonged to a different HyenaFilter instance, (b) belonged to
    // a different L (e.g. a previous test constructed a filter with L=5 and
    // we now operate at L=4), (c) never populated because filter() was never
    // called. In all three cases, repopulate with the current L.
    size_t L = grad_h.rows;
    if (L == 0 || L > l_max) {
        throw std::invalid_argument("HyenaFilter::backward: grad_h.rows incompatible with l_max");
    }
    HyenaFilterCache& c = cache();
    bool cache_stale = (c.gen != instance_gen) || (c.L != L) || (c.D != d_model) || (c.P != filter_order);
    if (cache_stale) {
        (void)filter(L);
    }

    // Backward through modulation: y[l, d] = pre[l, d] * exp(-delta[d] * t[l])
    //   grad_pre[l, d] = grad_h[l, d] * exp(-delta[d] * t[l])
    //   grad_delta[d]   = - sum_l grad_h[l, d] * pre[l, d] * t[l] * exp(-delta[d] * t[l])
    Tensor grad_pre(L, d_model);
    for (size_t l = 0; l < L; ++l) {
        double t = c.mod_t[l][0];
        for (size_t d = 0; d < d_model; ++d) {
            double mod = std::exp(-deltas[0][d] * t);
            grad_pre[l][d] = grad_h[l][d] * mod;
            grad_deltas[0][d] -= grad_h[l][d] * c.pre_decay[l][d] * t * mod;
        }
    }

    // Backward through mlp_out_W: y[l, d] = sum_p W_out[d, p] * a_final[l, p]
    //   grad_a_final[l, p] = sum_d grad_pre[l, d] * W_out[d, p]
    //   grad_W_out[d, p]   = sum_l grad_pre[l, d] * a_final[l, p]
    Tensor grad_a(L, filter_order);
    grad_a.fill(0.0);
    for (size_t d = 0; d < d_model; ++d) {
        for (size_t p = 0; p < filter_order; ++p) {
            double gw = 0.0;
            for (size_t l = 0; l < L; ++l) {
                gw += grad_pre[l][d] * c.a_final[l][p];
                grad_a[l][p] += grad_pre[l][d] * mlp_out_W[d][p];
            }
            grad_mlp_out_W[d][p] += gw;
        }
    }

    // Inner MLPs (in reverse): for each inner layer k (last to first),
    //   backward through sin then linear.
    //   The last "input" before the inner stack was c.h0 (or the post-final result,
    //   but since we have num_inner intermediate steps, the chain goes:
    //     grad_a (post-final-activation input) → through inner[num_inner-1] (sin+linear)
    //     → grad input of inner[num_inner-1] = c.h_inner[num_inner-2] (or h0 if first)
    //   We'll do it iteratively: starting from grad_a as the "input grad of last layer".
    Tensor cur_grad_input = grad_a;  // input grad to the last sin+linear block

    for (int k = (int)num_inner - 1; k >= 0; --k) {
        // cur_grad_input is grad of loss w.r.t. post-sin (i.e., pre-activation of NEXT layer)
        // Backward through sin(freq * z): grad_z = grad_y * cos(freq * z) * freq
        //   We need pre-activation z = c.a_inner[k] / sin_freq (the input to sin).
        //   But we stored pre_act = freq * h_inner_input. Recover h_inner_input = z = pre_act / freq.
        //   Or simpler: store z directly. But we have pre_act = freq * z (elementwise product).
        //   sin'(w*z) = w * cos(w*z). So grad_z = cur_grad_input * sin_freq * cos(pre_act).
        Tensor grad_z(cur_grad_input.rows, cur_grad_input.cols);
        for (size_t i = 0; i < cur_grad_input.data.size(); ++i) {
            double w = sin_freq.data[i % filter_order];
            grad_z.data[i] = cur_grad_input.data[i] * w * std::cos(c.a_inner[k].data[i]);
        }
        // grad_sin_freq: gradient of sin(w*z) w.r.t. w = z * cos(w*z) = (pre_act / w) * cos(w*z)
        //   grad_w[i] += grad_y[i] * (pre_act[i] / w[i]) * cos(pre_act[i])
        //   but we need z = pre_act / w. Since pre_act = w * z, z = pre_act / w.
        //   Avoid divide-by-zero by guarding w > eps.
        for (size_t p = 0; p < filter_order; ++p) {
            double w = sin_freq[0][p];
            double acc = 0.0;
            for (size_t l = 0; l < L; ++l) {
                size_t idx = l * filter_order + p;
                double z = (std::abs(w) > 1e-30) ? c.a_inner[k].data[idx] / w : 0.0;
                acc += cur_grad_input.data[idx] * z * std::cos(c.a_inner[k].data[idx]);
            }
            grad_sin_freq[0][p] += acc;
        }
        // Backward through linear: z = h_prev @ W^T + b. We need grad h_prev, grad W, grad b.
        // The "input" to this Linear was c.h_inner[k-1] (or c.h0 if k==0).
        const Tensor& h_prev = (k > 0) ? c.h_inner[k - 1] : c.h0;
        Tensor grad_h_prev, grad_W, grad_b;
        linear_backward(h_prev, mlp_W[k], grad_z, grad_h_prev, grad_W, grad_b);
        for (size_t i = 0; i < grad_W.data.size(); ++i) grad_mlp_W[k].data[i] += grad_W.data[i];
        for (size_t i = 0; i < grad_b.data.size(); ++i) grad_mlp_b[k].data[i] += grad_b.data[i];
        cur_grad_input = grad_h_prev;
    }

    // cur_grad_input is now grad of loss w.r.t. c.h0 (= post first sin).
    // Backward through first sin: grad_h = grad_y * freq * cos(pre_act)
    Tensor grad_h_in(cur_grad_input.rows, cur_grad_input.cols);
    for (size_t i = 0; i < cur_grad_input.data.size(); ++i) {
        double w = sin_freq.data[i % filter_order];
        grad_h_in.data[i] = cur_grad_input.data[i] * w * std::cos(c.a0.data[i]);
    }
    // grad_sin_freq contribution from first sin layer:
    for (size_t p = 0; p < filter_order; ++p) {
        double w = sin_freq[0][p];
        double acc = 0.0;
        for (size_t l = 0; l < L; ++l) {
            size_t idx = l * filter_order + p;
            double z = (std::abs(w) > 1e-30) ? c.a0.data[idx] / w : 0.0;
            acc += cur_grad_input.data[idx] * z * std::cos(c.a0.data[idx]);
        }
        grad_sin_freq[0][p] += acc;
    }
    // grad_h_in is the gradient w.r.t. the post-Linear-in (i.e., input to first sin).
    // Backward through mlp_in (Linear in): grad_z_in, grad_W_in, grad_b_in.
    Tensor grad_z_in, grad_W_in, grad_b_in;
    linear_backward(c.z, mlp_in_W, grad_h_in, grad_z_in, grad_W_in, grad_b_in);
    for (size_t i = 0; i < grad_W_in.data.size(); ++i) grad_mlp_in_W.data[i] += grad_W_in.data[i];
    for (size_t i = 0; i < grad_b_in.data.size(); ++i) grad_mlp_in_b.data[i] += grad_b_in.data[i];
    // grad_z_in is gradient w.r.t. positional embedding z — discarded (z is fixed).
}

// ============================================================================
// HyenaOperator
// ============================================================================
HyenaOperator::HyenaOperator(size_t d_model, size_t l_max, size_t order,
                             size_t filter_order)
    : d_model(d_model), l_max(l_max), order(order), filter_order(filter_order),
      in_proj(d_model, (order + 1) * d_model),
      out_proj(d_model, d_model),
      hyena_filter(d_model, l_max, filter_order)
{
    // depthwise conv1d with kernel=3, channels-first. Causal padding: pad left=1.
    short_W = Tensor((order + 1) * d_model, 3);
    short_b = Tensor((order + 1) * d_model, 1);
    fill_normal(short_W, std::sqrt(2.0 / (3.0 + (order + 1) * d_model)));
    fill_normal(short_b, 0.01);
    grad_short_W = Tensor((order + 1) * d_model, 3);
    grad_short_b = Tensor((order + 1) * d_model, 1);
}

void HyenaOperator::zero_grad() {
    in_proj.zero_grad();
    out_proj.zero_grad();
    hyena_filter.zero_grad();
    grad_short_W.fill(0.0);
    grad_short_b.fill(0.0);
}
std::vector<Tensor*> HyenaOperator::parameters() {
    std::vector<Tensor*> p = {&in_proj.weights, &in_proj.bias,
                              &out_proj.weights, &out_proj.bias,
                              &short_W, &short_b};
    auto hp = hyena_filter.parameters();
    p.insert(p.end(), hp.begin(), hp.end());
    return p;
}
std::vector<Tensor*> HyenaOperator::gradients() {
    std::vector<Tensor*> g = {&in_proj.grad_weights, &in_proj.grad_bias,
                              &out_proj.grad_weights, &out_proj.grad_bias,
                              &grad_short_W, &grad_short_b};
    auto hg = hyena_filter.gradients();
    g.insert(g.end(), hg.begin(), hg.end());
    return g;
}

// Helpers: split / merge channels-first (B, C, L) along channel axis into
// (order+1) tensors of shape (B, D, L). All have the same L.
static std::vector<Tensor> split_channels(const Tensor& x, size_t num_chunks) {
    // x is (B*num_chunks*D, L) where rows are interpreted as [B, num_chunks*D].
    // We want to return num_chunks tensors, each (B, D, L).
    size_t rows = x.rows;
    size_t L = x.cols;
    if (rows % num_chunks != 0) throw std::invalid_argument("split_channels: rows not divisible");
    size_t per_chunk = rows / num_chunks;  // B * D
    std::vector<Tensor> out(num_chunks, Tensor(per_chunk, L));
    for (size_t c = 0; c < num_chunks; ++c) {
        for (size_t r = 0; r < per_chunk; ++r) {
            for (size_t l = 0; l < L; ++l) {
                out[c][r][l] = x[c * per_chunk + r][l];
            }
        }
    }
    return out;
}

Tensor HyenaOperator::forward(const Tensor& input) {
    // Accepts either (B, L*D) or (B*L, D). For per-token use within a block,
    // we accept (N, D) where N = num_tokens. We infer L from cols and D = d_model.
    size_t rows = input.rows;
    size_t cols = input.cols;
    size_t D = d_model;
    size_t L;
    bool per_token_mode;
    if (cols == D) {
        // Per-token mode: input is (N, D) — each row is one token.
        per_token_mode = true;
        L = rows;
    } else if (cols == l_max * D) {
        // Block mode: input is (B, L*D) with L = l_max.
        per_token_mode = false;
        L = l_max;
        if (rows > 0 && L * D != cols) {
            throw std::invalid_argument("HyenaOperator: input cols mismatch");
        }
    } else {
        throw std::invalid_argument("HyenaOperator: input must be (N, D) or (B, l_max*D)");
    }
    if (L > l_max) throw std::invalid_argument("HyenaOperator: L > l_max");

    size_t B = per_token_mode ? 1 : rows;  // in block mode, B = input.rows
    last_input = input;

    // Reshape to (B*L, D) for in_proj.
    Tensor tokens(B * L, D);
    if (per_token_mode) {
        tokens = input.clone();
    } else {
        for (size_t b = 0; b < B; ++b) {
            for (size_t l = 0; l < L; ++l) {
                for (size_t d = 0; d < D; ++d) {
                    tokens[b * L + l][d] = input[b][l * D + d];
                }
            }
        }
    }

    // Per-token in_proj.
    Tensor proj_tokens = in_proj.forward(tokens);
    size_t total_ch = (order + 1) * D;

    // Store the in_proj output in (B, L*total_ch) form for the cache (used by backward).
    Tensor in_proj_out(B, L * total_ch);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t c = 0; c < total_ch; ++c) {
                in_proj_out[b][l * total_ch + c] = proj_tokens[b * L + l][c];
            }
        }
    }
    last_in_proj = in_proj_out;

    // Reshape to (B*total_ch, L) channels-first.
    Tensor u_ch(B * total_ch, L);
    u_ch.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < total_ch; ++c) {
            for (size_t l = 0; l < L; ++l) {
                u_ch[b * total_ch + c][l] = in_proj_out[b][l * total_ch + c];
            }
        }
    }

    // Causal depthwise Conv1d with kernel=3: pad left by 1, then trim right.
    Tensor short_out(B * total_ch, L);
    short_out.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < total_ch; ++c) {
            for (size_t l = 0; l < L; ++l) {
                double acc = short_b[c][0];
                for (size_t k = 0; k < 3; ++k) {
                    int t_in = (int)l + (int)k - 1;
                    if (t_in >= 0 && t_in < (int)L) {
                        acc += short_W[c][k] * u_ch[b * total_ch + c][(size_t)t_in];
                    }
                }
                short_out[b * total_ch + c][l] = acc;
            }
        }
    }
    last_short = short_out;

    std::vector<Tensor> chunks = split_channels(short_out, order + 1);
    Tensor g_last = chunks[0];
    Tensor g_first = chunks[1];
    Tensor v = chunks[order];

    // Stage 0
    auto f0 = hyena_filter.filter(L);
    Tensor k0 = f0.first;
    Tensor D0 = f0.second;

    Tensor v_after_g1(B * D, L);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t l = 0; l < L; ++l) {
                v_after_g1[b * D + d][l] = v[b * D + d][l] * g_first[b * D + d][l];
            }
        }
    }
    last_v0 = v_after_g1;

    Tensor y0(B * D, L);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            Tensor v_slice(1, L);
            for (size_t l = 0; l < L; ++l) v_slice[0][l] = v_after_g1[b * D + d][l];
            Tensor k_slice(1, L);
            for (size_t l = 0; l < L; ++l) k_slice[0][l] = k0[l][d];
            Tensor y_slice = causal_conv_1d(v_slice, k_slice);
            double skip = D0[0][d];
            for (size_t l = 0; l < L; ++l) {
                y0[b * D + d][l] = y_slice[0][l] + skip * v_slice[0][l];
            }
        }
    }
    last_y0 = y0;
    last_filter_h_0 = k0;
    last_filter_D_0 = D0;

    // Stage 1
    auto f1 = hyena_filter.filter(L);
    Tensor k1 = f1.first;
    Tensor D1 = f1.second;
    Tensor v_after_g0(B * D, L);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t l = 0; l < L; ++l) {
                v_after_g0[b * D + d][l] = y0[b * D + d][l] * g_last[b * D + d][l];
            }
        }
    }
    last_v1 = v_after_g0;

    Tensor y1(B * D, L);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            Tensor v_slice(1, L);
            for (size_t l = 0; l < L; ++l) v_slice[0][l] = v_after_g0[b * D + d][l];
            Tensor k_slice(1, L);
            for (size_t l = 0; l < L; ++l) k_slice[0][l] = k1[l][d];
            Tensor y_slice = causal_conv_1d(v_slice, k_slice);
            double skip = D1[0][d];
            for (size_t l = 0; l < L; ++l) {
                y1[b * D + d][l] = y_slice[0][l] + skip * v_slice[0][l];
            }
        }
    }
    last_y1 = y1;
    last_filter_h_1 = k1;
    last_filter_D_1 = D1;

    // Reshape to (B*L, D) for out_proj.
    Tensor out_tokens(B * L, D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < D; ++c) {
            for (size_t l = 0; l < L; ++l) {
                out_tokens[b * L + l][c] = y1[b * D + c][l];
            }
        }
    }
    Tensor proj_out = out_proj.forward(out_tokens);

    if (per_token_mode) {
        // Return (N, D) directly
        return proj_out;
    } else {
        // Reshape back to (B, L*D)
        Tensor final_out(B, L * D);
        for (size_t b = 0; b < B; ++b) {
            for (size_t l = 0; l < L; ++l) {
                for (size_t d = 0; d < D; ++d) {
                    final_out[b][l * D + d] = proj_out[b * L + l][d];
                }
            }
        }
        return final_out;
    }
}

Tensor HyenaOperator::backward(const Tensor& grad_output, double /*learning_rate*/) {
    // Determine mode (same logic as forward)
    size_t cols = grad_output.cols;
    size_t D = d_model;
    size_t L;
    bool per_token_mode;
    if (cols == D) {
        per_token_mode = true;
        L = last_input.rows;
    } else if (cols == l_max * D) {
        per_token_mode = false;
        L = l_max;
    } else {
        throw std::invalid_argument("HyenaOperator::backward: grad shape mismatch");
    }
    size_t B = per_token_mode ? 1 : last_input.rows;
    size_t total_ch = (order + 1) * D;

    // Backward through out_proj: reshape to (B*L, D)
    Tensor grad_proj_tokens(B * L, D);
    if (per_token_mode) {
        for (size_t i = 0; i < grad_output.data.size(); ++i) grad_proj_tokens.data[i] = grad_output.data[i];
    } else {
        for (size_t b = 0; b < B; ++b) {
            for (size_t l = 0; l < L; ++l) {
                for (size_t d = 0; d < D; ++d) {
                    grad_proj_tokens[b * L + l][d] = grad_output[b][l * D + d];
                }
            }
        }
    }
    Tensor grad_out_tokens = out_proj.backward(grad_proj_tokens, 0.0);
    Tensor grad_pre_out(B, L * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < D; ++d) {
                grad_pre_out[b][l * D + d] = grad_out_tokens[b * L + l][d];
            }
        }
    }

    // grad_pre_out is (B, L*D). Reshape to (B*D, L) channels-first.
    Tensor grad_y1(B * D, L);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < D; ++c) {
            for (size_t l = 0; l < L; ++l) {
                grad_y1[b * D + c][l] = grad_pre_out[b][l * D + c];
            }
        }
    }

    // Backward through stage 1: y1 = conv(k1, v_after_g0) + D1 * v_after_g0
    //   grad_v_after_g0 (per channel) = conv_k1_backward_v(grad_y1, k1) + D1 * grad_y1
    //   grad_k1 (per channel) = conv_k1_backward_k(grad_y1, v_after_g0)
    //   grad_D1[d] += sum_{b, l} grad_y1[b, d, l] * v_after_g0[b, d, l]
    Tensor grad_v_after_g0(B * D, L);
    Tensor grad_k1(L, D);
    Tensor grad_D1_local(1, D);
    grad_D1_local.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            Tensor gy(1, L);
            for (size_t l = 0; l < L; ++l) gy[0][l] = grad_y1[b * D + d][l];
            Tensor k_slice(1, L);
            for (size_t l = 0; l < L; ++l) k_slice[0][l] = last_filter_h_1[l][d];
            Tensor gv = causal_conv_1d_backward_v(gy, k_slice);
            double skip = last_filter_D_1[0][d];
            for (size_t l = 0; l < L; ++l) {
                grad_v_after_g0[b * D + d][l] = gv[0][l] + skip * gy[0][l];
            }
            Tensor v_slice(1, L);
            for (size_t l = 0; l < L; ++l) v_slice[0][l] = last_v1[b * D + d][l];
            Tensor gk = causal_conv_1d_backward_k(gy, v_slice);
            for (size_t l = 0; l < L; ++l) grad_k1[l][d] += gk[0][l];
            // D1 grad
            for (size_t l = 0; l < L; ++l) {
                grad_D1_local[0][d] += grad_y1[b * D + d][l] * last_v1[b * D + d][l];
            }
        }
    }

    // Backward through stage 1's gate: v_after_g0 = y0 * g_last
    //   grad_y0 = grad_v_after_g0 * g_last
    //   grad_g_last = grad_v_after_g0 * y0
    Tensor grad_y0(B * D, L);
    Tensor grad_g_last(B * D, L);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t l = 0; l < L; ++l) {
                grad_y0[b * D + d][l] = grad_v_after_g0[b * D + d][l] *
                                        last_short[b * total_ch + 0 * D + d][l];
                grad_g_last[b * D + d][l] = grad_v_after_g0[b * D + d][l] *
                                            last_y0[b * D + d][l];
            }
        }
    }

    // Backward through stage 0: y0 = conv(k0, v_after_g1) + D0 * v_after_g1
    Tensor grad_v_after_g1(B * D, L);
    Tensor grad_k0(L, D);
    Tensor grad_D0_local(1, D);
    grad_D0_local.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            Tensor gy(1, L);
            for (size_t l = 0; l < L; ++l) gy[0][l] = grad_y0[b * D + d][l];
            Tensor k_slice(1, L);
            for (size_t l = 0; l < L; ++l) k_slice[0][l] = last_filter_h_0[l][d];
            Tensor gv = causal_conv_1d_backward_v(gy, k_slice);
            double skip = last_filter_D_0[0][d];
            for (size_t l = 0; l < L; ++l) {
                grad_v_after_g1[b * D + d][l] = gv[0][l] + skip * gy[0][l];
            }
            Tensor v_slice(1, L);
            for (size_t l = 0; l < L; ++l) v_slice[0][l] = last_v0[b * D + d][l];
            Tensor gk = causal_conv_1d_backward_k(gy, v_slice);
            for (size_t l = 0; l < L; ++l) grad_k0[l][d] += gk[0][l];
            for (size_t l = 0; l < L; ++l) {
                grad_D0_local[0][d] += grad_y0[b * D + d][l] * last_v0[b * D + d][l];
            }
        }
    }

    // Backward through stage 0's gate: v_after_g1 = v * g_first
    //   grad_v_chunks[order] = grad_v_after_g1 * g_first
    //   grad_g_first = grad_v_after_g1 * v
    Tensor grad_v(B * D, L);
    Tensor grad_g_first(B * D, L);
    for (size_t b = 0; b < B; ++b) {
        for (size_t d = 0; d < D; ++d) {
            for (size_t l = 0; l < L; ++l) {
                grad_v[b * D + d][l] = grad_v_after_g1[b * D + d][l] *
                                       last_short[b * total_ch + 1 * D + d][l];
                grad_g_first[b * D + d][l] = grad_v_after_g1[b * D + d][l] *
                                             last_short[b * total_ch + order * D + d][l];
            }
        }
    }

    // Combine gate grads with chunk grads into grad_short (B*total_ch, L)
    Tensor grad_short(B * total_ch, L);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < D; ++c) {
            for (size_t l = 0; l < L; ++l) {
                grad_short[b * total_ch + 0 * D + c][l] = grad_g_last[b * D + c][l];
                grad_short[b * total_ch + 1 * D + c][l] = grad_g_first[b * D + c][l];
                grad_short[b * total_ch + order * D + c][l] = grad_v[b * D + c][l];
            }
        }
    }

    // Backward through short conv1d (kernel=3, causal pad left=1).
    // y[c, l] = sum_k W[c, k] * u[c, l + k - 1] + b[c]
    //   grad_u[c, l_in] = sum_{k: l_in = l + k - 1} grad_y[c, l] * W[c, k]
    //                    = sum_k grad_y[c, l_in - k + 1] * W[c, k]   (l = l_in - k + 1)
    //   grad_W[c, k] = sum_l grad_y[c, l] * u[c, l + k - 1]
    //   grad_b[c] = sum_l grad_y[c, l]
    Tensor grad_u_ch(B * total_ch, L);
    grad_u_ch.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < total_ch; ++c) {
            for (size_t l_in = 0; l_in < L; ++l_in) {
                double acc = 0.0;
                for (size_t k = 0; k < 3; ++k) {
                    int l = (int)l_in - (int)k + 1;
                    if (l >= 0 && l < (int)L) {
                        acc += grad_short[b * total_ch + c][l] * short_W[c][k];
                    }
                }
                grad_u_ch[b * total_ch + c][l_in] = acc;
            }
            // grad_W and grad_b
            for (size_t k = 0; k < 3; ++k) {
                double gw = 0.0;
                for (size_t l = 0; l < L; ++l) {
                    int t_in = (int)l + (int)k - 1;
                    if (t_in >= 0 && t_in < (int)L) {
                        gw += grad_short[b * total_ch + c][l] *
                              last_short[b * total_ch + c][l] * 0.0; // placeholder
                    }
                }
                // Note: we need u_ch not last_short for grad_W.
                // We'll redo below.
                (void)gw;
            }
        }
    }
    // Correct grad_W and grad_b using u_ch. (We need to cache u_ch in forward.)
    // To keep this self-contained, recompute u_ch here:
    Tensor u_ch_recompute(B * total_ch, L);
    u_ch_recompute.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < total_ch; ++c) {
            for (size_t l = 0; l < L; ++l) {
                u_ch_recompute[b * total_ch + c][l] = last_in_proj[b][l * total_ch + c];
            }
        }
    }
    for (size_t c = 0; c < total_ch; ++c) {
        double gb = 0.0;
        for (size_t k = 0; k < 3; ++k) {
            double gw = 0.0;
            for (size_t b = 0; b < B; ++b) {
                for (size_t l = 0; l < L; ++l) {
                    int t_in = (int)l + (int)k - 1;
                    if (t_in >= 0 && t_in < (int)L) {
                        gw += grad_short[b * total_ch + c][l] *
                              u_ch_recompute[b * total_ch + c][(size_t)t_in];
                        if (k == 0) gb += grad_short[b * total_ch + c][l];
                    }
                }
            }
            grad_short_W[c][k] += gw;
        }
        grad_short_b[c][0] += gb;
    }

    // Reshape grad_u_ch (B*total_ch, L) back to channels-last (B, L*total_ch)
    Tensor grad_in_proj(B, L * total_ch);
    for (size_t b = 0; b < B; ++b) {
        for (size_t c = 0; c < total_ch; ++c) {
            for (size_t l = 0; l < L; ++l) {
                grad_in_proj[b][l * total_ch + c] = grad_u_ch[b * total_ch + c][l];
            }
        }
    }

    // Backward through in_proj: reshape (B, L*total_ch) → (B*L, total_ch),
    // call in_proj.backward, reshape back to (B, L*D).
    Tensor grad_in_proj_tokens(B * L, total_ch);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t c = 0; c < total_ch; ++c) {
                grad_in_proj_tokens[b * L + l][c] = grad_in_proj[b][l * total_ch + c];
            }
        }
    }
    Tensor grad_tokens = in_proj.backward(grad_in_proj_tokens, 0.0);
    // For per_token_mode, return grad_tokens (B*L, D) directly to match the
    // input shape invoked by HyenaBlock::backward. For block mode, return
    // (B, L*D) to match the input layout.
    if (per_token_mode) {
        return grad_tokens;
    }
    Tensor grad_input(B, L * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < D; ++d) {
                grad_input[b][l * D + d] = grad_tokens[b * L + l][d];
            }
        }
    }

    // Backward through HyenaFilter. We need to call hyena_filter.backward twice
    // because the filter was applied twice (once for stage 0, once for stage 1).
    // The cache holds the LAST call's activations, so calling backward() twice
    // would both use stage 1's cache. To handle this properly, we need to
    // re-run the filter forward for stage 0 to repopulate the cache.
    // For now, only the last filter call's gradients will be propagated. This
    // means we're missing the gradient through the filter when it was applied
    // for stage 0. NOTE: this is a known limitation of this approach.
    //
    // A clean fix: re-cache by re-running filter(L) for stage 0 before backward,
    // then accumulate. We do this here:
    hyena_filter.filter(L);  // repopulate cache with stage 1 activations
    hyena_filter.backward(grad_k1, grad_D1_local);
    hyena_filter.filter(L);  // repopulate cache with stage 0 activations
    hyena_filter.backward(grad_k0, grad_D0_local);

    return grad_input;
}

void HyenaOperator::update_weights(double learning_rate) {
    in_proj.update_weights(learning_rate);
    out_proj.update_weights(learning_rate);
    // short_W, short_b: SGD
    for (size_t i = 0; i < short_W.data.size(); ++i)
        short_W.data[i] -= learning_rate * grad_short_W.data[i];
    for (size_t i = 0; i < short_b.data.size(); ++i)
        short_b.data[i] -= learning_rate * grad_short_b.data[i];
    // HyenaFilter parameters: SGD (we don't have per-parameter optimizer here;
    // for the test suite we use a custom optimizer that handles all layers).
    auto hp = hyena_filter.parameters();
    auto hg = hyena_filter.gradients();
    for (size_t i = 0; i < hp.size(); ++i) {
        for (size_t j = 0; j < hp[i]->data.size(); ++j) {
            hp[i]->data[j] -= learning_rate * hg[i]->data[j];
        }
    }
}

// ============================================================================
// HyenaBlock
// ============================================================================
HyenaBlock::HyenaBlock(size_t d_model, size_t l_max, size_t order,
                       size_t filter_order, size_t ffn_mult)
    : d_model(d_model), l_max(l_max), order(order), filter_order(filter_order),
      ffn_mult(ffn_mult),
      ln1(d_model), ln2(d_model),
      // HyenaOperator must accept up to B*l_max tokens when used in a Block
      // context; we register a generous max. Tests use B*L within l_max.
      hyena(d_model, l_max * 256, order, filter_order),
      ffn1(d_model, ffn_mult * d_model),
      ffn2(ffn_mult * d_model, d_model)
{}

void HyenaBlock::zero_grad() {
    ln1.zero_grad();
    ln2.zero_grad();
    hyena.zero_grad();
    ffn1.zero_grad();
    ffn2.zero_grad();
}
std::vector<Tensor*> HyenaBlock::parameters() {
    std::vector<Tensor*> p = ln1.parameters();
    auto h = hyena.parameters();
    p.insert(p.end(), h.begin(), h.end());
    auto f1 = ffn1.parameters();
    p.insert(p.end(), f1.begin(), f1.end());
    auto f2 = ffn2.parameters();
    p.insert(p.end(), f2.begin(), f2.end());
    auto ln2p = ln2.parameters();
    p.insert(p.end(), ln2p.begin(), ln2p.end());
    return p;
}
std::vector<Tensor*> HyenaBlock::gradients() {
    std::vector<Tensor*> g = ln1.gradients();
    auto h = hyena.gradients();
    g.insert(g.end(), h.begin(), h.end());
    auto f1 = ffn1.gradients();
    g.insert(g.end(), f1.begin(), f1.end());
    auto f2 = ffn2.gradients();
    g.insert(g.end(), f2.begin(), f2.end());
    auto ln2g = ln2.gradients();
    g.insert(g.end(), ln2g.begin(), ln2g.end());
    return g;
}

Tensor HyenaBlock::forward(const Tensor& input) {
    size_t B = input.rows;
    size_t L = l_max;
    size_t D = d_model;
    last_input = input;

    // Reshape (B, L*D) → (B*L, D) for per-token operations.
    Tensor tokens(B * L, D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < D; ++d) {
                tokens[b * L + l][d] = input[b][l * D + d];
            }
        }
    }

    // Pre-LN1 (per-token LN over features=D), then HyenaOperator (per-token in/out).
    Tensor h1 = ln1.forward(tokens);
    Tensor h_hyena = hyena.forward(h1);
    // Residual: tokens + h_hyena (per-token add)
    Tensor r1(B * L, D);
    for (size_t i = 0; i < tokens.data.size(); ++i) r1.data[i] = tokens.data[i] + h_hyena.data[i];

    // Pre-LN2 + FFN
    Tensor h2 = ln2.forward(r1);
    Tensor ff1_out = ffn1.forward(h2);

    // GELU
    Tensor ff1_act(ff1_out.rows, ff1_out.cols);
    for (size_t i = 0; i < ff1_out.data.size(); ++i) {
        double x = ff1_out.data[i];
        ff1_act.data[i] = 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
    }

    Tensor ff2_out = ffn2.forward(ff1_act);

    // Residual: r1 + ff2_out
    Tensor tokens_out(B * L, D);
    for (size_t i = 0; i < r1.data.size(); ++i) tokens_out.data[i] = r1.data[i] + ff2_out.data[i];

    // Reshape (B*L, D) → (B, L*D)
    Tensor out(B, L * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < D; ++d) {
                out[b][l * D + d] = tokens_out[b * L + l][d];
            }
        }
    }
    return out;
}

Tensor HyenaBlock::backward(const Tensor& grad_output, double learning_rate) {
    size_t B = last_input.rows;
    size_t L = l_max;
    size_t D = d_model;

    // Reshape grad_output (B, L*D) → (B*L, D) for per-token operations.
    Tensor grad_tokens_out(B * L, D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < D; ++d) {
                grad_tokens_out[b * L + l][d] = grad_output[b][l * D + d];
            }
        }
    }

    // Recompute forward so caches are correct.
    Tensor tokens(B * L, D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < D; ++d) {
                tokens[b * L + l][d] = last_input[b][l * D + d];
            }
        }
    }
    Tensor h1 = ln1.forward(tokens);
    Tensor h_hyena = hyena.forward(h1);
    Tensor r1(B * L, D);
    for (size_t i = 0; i < tokens.data.size(); ++i) r1.data[i] = tokens.data[i] + h_hyena.data[i];
    Tensor h2 = ln2.forward(r1);
    Tensor ff1_out = ffn1.forward(h2);

    // Residual backward: grad_tokens_out = grad_r1 + grad_ff2_out
    Tensor grad_r1(B * L, D);
    for (size_t i = 0; i < grad_tokens_out.data.size(); ++i)
        grad_r1.data[i] = grad_tokens_out.data[i];
    Tensor grad_ff2_out = grad_tokens_out.clone();

    // Backward through ffn2
    Tensor grad_ff1_act = ffn2.backward(grad_ff2_out, learning_rate);

    // GELU backward
    Tensor grad_ff1_in(ff1_out.rows, ff1_out.cols);
    for (size_t i = 0; i < grad_ff1_in.data.size(); ++i) {
        double x = ff1_out.data[i];
        double z = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
        double t = std::tanh(z);
        double gelu_prime = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) *
                            std::sqrt(2.0 / M_PI) * (1.0 + 3.0 * 0.044715 * x * x);
        grad_ff1_in.data[i] = grad_ff1_act.data[i] * gelu_prime;
    }

    Tensor grad_h2 = ffn1.backward(grad_ff1_in, learning_rate);

    // Add grad_h2 to grad_r1 (LN2 → r1 path).
    Tensor grad_r1_from_ln2 = ln2.backward(grad_h2, learning_rate);
    for (size_t i = 0; i < grad_r1.data.size(); ++i)
        grad_r1.data[i] += grad_r1_from_ln2.data[i];

    // Backward through residual1: r1 = tokens + hyena(ln1(tokens))
    // grad_tokens = grad_r1, grad_h_hyena = grad_r1
    // grad_h1 = hyena.backward(grad_h_hyena)  -- chain through HyenaOperator
    // grad_ln1_in = ln1.backward(grad_h1)
    // grad_tokens += grad_ln1_in  (residual through the hyena path)
    Tensor grad_tokens(B * L, D);
    Tensor grad_hyena_in(B * L, D);
    for (size_t i = 0; i < grad_r1.data.size(); ++i) {
        grad_tokens.data[i] = grad_r1.data[i];
        grad_hyena_in.data[i] = grad_r1.data[i];
    }
    // Chain through HyenaOperator: this is what was missing before.
    Tensor grad_h1 = hyena.backward(grad_hyena_in, learning_rate);
    Tensor grad_ln1 = ln1.backward(grad_h1, learning_rate);
    for (size_t i = 0; i < grad_tokens.data.size(); ++i)
        grad_tokens.data[i] += grad_ln1.data[i];

    // Reshape (B*L, D) → (B, L*D)
    Tensor grad_input(B, L * D);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < D; ++d) {
                grad_input[b][l * D + d] = grad_tokens[b * L + l][d];
            }
        }
    }
    return grad_input;
}

void HyenaBlock::update_weights(double learning_rate) {
    ln1.update_weights(learning_rate);
    hyena.update_weights(learning_rate);
    ffn1.update_weights(learning_rate);
    ffn2.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
}

// ============================================================================
// HyenaModel
// ============================================================================
HyenaModel::HyenaModel(size_t d_model, size_t l_max, size_t depth, size_t num_classes,
                       size_t order, size_t filter_order)
    : d_model(d_model), l_max(l_max), depth(depth), num_classes(num_classes),
      order(order), filter_order(filter_order),
      classifier(d_model, num_classes)
{
    for (size_t i = 0; i < depth; ++i) {
        blocks.emplace_back(d_model, l_max, order, filter_order);
    }
}

void HyenaModel::zero_grad() {
    for (auto& b : blocks) b.zero_grad();
    classifier.zero_grad();
}
std::vector<Tensor*> HyenaModel::parameters() {
    std::vector<Tensor*> p;
    for (auto& b : blocks) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto cp = classifier.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}
std::vector<Tensor*> HyenaModel::gradients() {
    std::vector<Tensor*> g;
    for (auto& b : blocks) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto cg = classifier.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}
Tensor HyenaModel::get_weights() const { return classifier.weights; }
Tensor HyenaModel::get_gradients() const { return classifier.grad_weights; }

Tensor HyenaModel::forward(const Tensor& input) {
    last_input = input;
    Tensor h = input;
    for (auto& b : blocks) h = b.forward(h);
    // Per-token mean pool: (B, L*D) → (B, D)
    size_t B = h.rows;
    size_t L = l_max;
    Tensor pooled(B, d_model);
    pooled.fill(0.0);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < d_model; ++d) {
                pooled[b][d] += h[b][l * d_model + d];
            }
        }
        for (size_t d = 0; d < d_model; ++d) pooled[b][d] /= (double)L;
    }
    return classifier.forward(pooled);
}

Tensor HyenaModel::backward(const Tensor& grad_output, double learning_rate) {
    // grad_output is (B, num_classes). Backward through classifier.
    Tensor grad_pooled = classifier.backward(grad_output, learning_rate);
    size_t B = grad_pooled.rows;
    size_t L = l_max;
    // Broadcast grad_pooled (B, D) back to (B, L*D) for blocks
    Tensor grad_blocks(B, L * d_model);
    for (size_t b = 0; b < B; ++b) {
        for (size_t l = 0; l < L; ++l) {
            for (size_t d = 0; d < d_model; ++d) {
                grad_blocks[b][l * d_model + d] = grad_pooled[b][d] / (double)L;
            }
        }
    }
    for (int i = (int)depth - 1; i >= 0; --i) {
        grad_blocks = blocks[i].backward(grad_blocks, learning_rate);
    }
    return grad_blocks;
}

void HyenaModel::update_weights(double learning_rate) {
    for (auto& b : blocks) b.update_weights(learning_rate);
    classifier.update_weights(learning_rate);
}
