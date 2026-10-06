#include "multi_head_hyena.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// ============================================================================
// MultiHeadHyena (MultiHyena) — Massaroli, Poli et al., "Laughing Hyena
// Distillery", arXiv:2310.18780 §4.
//
// Per head m (head_dim N = d_model / num_heads):
//
//   z^m[l][i*N+j] = k^m[l][i] * v^m[l][j]                  outer product
//   w^m[l][c]    = Σ_{s<=l} h^m[s] * z^m[s][c]             long convolution
//   y^m[l][i]    = Σ_j w^m[l][i*N+j] * q^m[l][i]           Hadamard with q
//   y = concat_m y^m                                       concatenation
//
// THE SHARED FILTER. §4 opens by asking "can we reduce the total number of
// filters without loss in quality?" and answers yes: `h^m` is ONE filter of
// width L applied to ALL N² elements of the outer-product plane independently
// ("apply a long convolution with filter h^m to all N×N elements
// independently"). So h^m[l][c] is the same value for every c, and the whole
// M-head layer uses M filters instead of N²·M. This is the layer's thesis and
// the test suite pins it.
//
// Because the shared filter has ONE column per timestep, each head's filter
// GENERATOR is a `HyenaFilter` built with d_model = 1 — it emits exactly one
// column h_gen ∈ R^L, which is then broadcast across the N² plane.
//
// Backward is the exact adjoint:
//
//   y^m[l][i] = (Σ_j w^m[l][i*N+j]) * q^m[l][i]
//     ⇒ dw^m[l][i*N+j] = dy^m[l][i] * q^m[l][i]
//     ⇒ dq^m[l][i]     = Σ_j dw^m[l][i*N+j]
//   conv backward:  dw → dz (input) and dh (filter)
//   outer backward: z = k ⊗ v
//     ⇒ dk^m[l][i] = Σ_j dz^m[l][i*N+j] * v^m[l][j]
//     ⇒ dv^m[l][j] = Σ_i dz^m[l][i*N+j] * k^m[l][i]
// ============================================================================

namespace {

// Causal convolution: y[t] = Σ_{s<=t} k[s] * v[t-s]. Both (1, L).
Tensor causal_conv(const Tensor& v, const Tensor& k) {
    size_t L = v.cols;
    Tensor y(1, L);
    y.fill(0.0);
    for (size_t t = 0; t < L; ++t) {
        double acc = 0.0;
        for (size_t s = 0; s <= t; ++s) acc += k[0][s] * v[0][t - s];
        y[0][t] = acc;
    }
    return y;
}

// dL/dv[t'] = Σ_{t>=t'} grad_y[t] * k[t - t']
Tensor causal_conv_grad_v(const Tensor& grad_y, const Tensor& k) {
    size_t L = grad_y.cols;
    Tensor gv(1, L);
    gv.fill(0.0);
    for (size_t tp = 0; tp < L; ++tp) {
        double acc = 0.0;
        for (size_t t = tp; t < L; ++t) acc += grad_y[0][t] * k[0][t - tp];
        gv[0][tp] = acc;
    }
    return gv;
}

// dL/dk[s] = Σ_{t>=s} grad_y[t] * v[t - s]
Tensor causal_conv_grad_k(const Tensor& grad_y, const Tensor& v) {
    size_t L = grad_y.cols;
    Tensor gk(1, L);
    gk.fill(0.0);
    for (size_t s = 0; s < L; ++s) {
        double acc = 0.0;
        for (size_t t = s; t < L; ++t) acc += grad_y[0][t] * v[0][t - s];
        gk[0][s] = acc;
    }
    return gk;
}

}  // namespace

// ============================================================================
// MultiHeadHyenaOperator
// ============================================================================
MultiHeadHyenaOperator::MultiHeadHyenaOperator(size_t d_model, size_t seq_len,
                                               size_t num_heads,
                                               size_t filter_order)
    : d_model_(d_model), seq_len_(seq_len), num_heads_(num_heads),
      filter_order_(filter_order),
      in_proj_(d_model, 3 * d_model),
      out_proj_(d_model, d_model)
{
    if (d_model == 0)
        throw std::invalid_argument("MultiHeadHyenaOperator: d_model must be > 0");
    if (seq_len == 0)
        throw std::invalid_argument("MultiHeadHyenaOperator: seq_len must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("MultiHeadHyenaOperator: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument(
            "MultiHeadHyenaOperator: d_model must be divisible by num_heads");
    if (filter_order == 0)
        throw std::invalid_argument("MultiHeadHyenaOperator: filter_order must be > 0");
    head_dim_ = d_model / num_heads;

    // One single-column filter generator per head (see file header).
    filters_.reserve(num_heads);
    last_h_.assign(num_heads, Tensor());
    pinned_.assign(num_heads, false);
    for (size_t m = 0; m < num_heads; ++m)
        filters_.push_back(std::make_unique<HyenaFilter>(1, seq_len, filter_order));
}

Tensor MultiHeadHyenaOperator::last_filter(size_t head) const {
    if (head >= num_heads_)
        throw std::out_of_range("last_filter: head out of range");
    return last_h_[head];
}

Tensor MultiHeadHyenaOperator::last_outer(size_t head) const {
    if (head >= num_heads_)
        throw std::out_of_range("last_outer: head out of range");
    return last_z_[head];
}

Tensor MultiHeadHyenaOperator::last_conv(size_t head) const {
    if (head >= num_heads_)
        throw std::out_of_range("last_conv: head out of range");
    return last_w_[head];
}

void MultiHeadHyenaOperator::set_filter_for_test(size_t head, const Tensor& h) {
    if (head >= num_heads_)
        throw std::out_of_range("set_filter_for_test: head out of range");
    if (h.rows != seq_len_ || h.cols != head_dim_ * head_dim_)
        throw std::invalid_argument("set_filter_for_test: h must be (seq_len, N*N)");
    pinned_[head] = true;
    override_h_ = h;   // used verbatim on the next forward for this head
}

Tensor MultiHeadHyenaOperator::forward(const Tensor& input) {
    if (input.cols != d_model_)
        throw std::invalid_argument("MultiHeadHyenaOperator: input must be (L, d_model)");
    if (input.rows != seq_len_)
        throw std::invalid_argument("MultiHeadHyenaOperator: input rows must equal seq_len");
    last_input_ = input;

    Tensor proj = in_proj_.forward(input);   // (L, 3*d_model)
    Tensor q(input.rows, d_model_), k(input.rows, d_model_), v(input.rows, d_model_);
    for (size_t l = 0; l < input.rows; ++l)
        for (size_t d = 0; d < d_model_; ++d) {
            q[l][d] = proj[l][d];
            k[l][d] = proj[l][d_model_ + d];
            v[l][d] = proj[l][2 * d_model_ + d];
        }
    return forward_from_qkv(q, k, v);
}

Tensor MultiHeadHyenaOperator::forward_from_qkv(const Tensor& q, const Tensor& k,
                                                const Tensor& v) {
    if (q.rows != k.rows || q.rows != v.rows || q.cols != d_model_ ||
        k.cols != d_model_ || v.cols != d_model_)
        throw std::invalid_argument("forward_from_qkv: q/k/v must all be (L, d_model)");
    size_t L = q.rows;

    // Cache q,k,v stacked (rows [0,L)=q, [L,2L)=k, [2L,3L)=v) so backward has
    // them without re-running in_proj.
    last_qkv_ = Tensor(3 * L, d_model_);
    for (size_t l = 0; l < L; ++l)
        for (size_t d = 0; d < d_model_; ++d) {
            last_qkv_[l][d] = q[l][d];
            last_qkv_[L + l][d] = k[l][d];
            last_qkv_[2 * L + l][d] = v[l][d];
        }

    const size_t N = head_dim_;
    const size_t P = N * N;

    last_z_.assign(num_heads_, Tensor());
    last_w_.assign(num_heads_, Tensor());
    last_qh_.assign(num_heads_, Tensor());

    Tensor concat(L, d_model_);
    concat.fill(0.0);

    for (size_t m = 0; m < num_heads_; ++m) {
        // --- outer product: z[l][i*N+j] = k[l][m*N+i] * v[l][m*N+j]
        Tensor z(L, P);
        z.fill(0.0);
        for (size_t l = 0; l < L; ++l)
            for (size_t i = 0; i < N; ++i) {
                const double ki = k[l][m * N + i];
                for (size_t j = 0; j < N; ++j) z[l][i * N + j] = ki * v[l][m * N + j];
            }
        last_z_[m] = z;

        // --- the shared filter h^m, broadcast across the whole N×N plane
        Tensor h(L, P);
        if (pinned_[m]) {
            h = override_h_;
        } else {
            auto hp = filters_[m]->filter(L);   // (h_gen (L,1), D_skip (1,1))
            const Tensor& hgen = hp.first;      // (L, 1) — the single column
            h.fill(0.0);
            for (size_t l = 0; l < L; ++l)
                for (size_t c = 0; c < P; ++c) h[l][c] = hgen[l][0];
        }
        last_h_[m] = h;

        // --- long convolution, independently per outer-product channel c
        Tensor w(L, P);
        w.fill(0.0);
        for (size_t c = 0; c < P; ++c) {
            Tensor vs(1, L), hs(1, L);
            for (size_t l = 0; l < L; ++l) { vs[0][l] = z[l][c]; hs[0][l] = h[l][c]; }
            Tensor ys = causal_conv(vs, hs);
            for (size_t l = 0; l < L; ++l) w[l][c] = ys[0][l];
        }
        last_w_[m] = w;

        // --- Hadamard with q^m, written into the concatenated output
        Tensor qh(L, N);
        qh.fill(0.0);
        for (size_t l = 0; l < L; ++l)
            for (size_t i = 0; i < N; ++i) {
                const double qi = q[l][m * N + i];
                double acc = 0.0;
                for (size_t j = 0; j < N; ++j) acc += w[l][i * N + j] * qi;
                qh[l][i] = acc;
                concat[l][m * N + i] = acc;
            }
        last_qh_[m] = qh;
    }

    return out_proj_.forward(concat);
}

Tensor MultiHeadHyenaOperator::backward(const Tensor& grad_output, double /*lr*/) {
    size_t L = seq_len_;
    const size_t N = head_dim_;
    const size_t P = N * N;

    Tensor grad_concat = out_proj_.backward(grad_output, 0.0);

    Tensor dq(L, d_model_), dk(L, d_model_), dv(L, d_model_);
    dq.fill(0.0); dk.fill(0.0); dv.fill(0.0);

    const Tensor& qkv = last_qkv_;

    for (size_t m = 0; m < num_heads_; ++m) {
        const Tensor& z = last_z_[m];
        const Tensor& h = last_h_[m];

        // y^l[i] = q^l[i] · S^l[i]   where  S^l[i] = Σ_j w^l[i*N+j]
        //   ⇒ dw^l[i*N+j] = dy^l[i] * q^l[i]
        //   ⇒ dq^l[i]     = dy^l[i] * S^l[i]
        // Note dq is NOT Σ_j dw — that would be N·dy·q, off by both a factor
        // N and the w/q substitution. last_w_[m] holds S.
        Tensor dw(L, P);
        dw.fill(0.0);
        for (size_t l = 0; l < L; ++l)
            for (size_t i = 0; i < N; ++i) {
                const double gy_ = grad_concat[l][m * N + i];
                const double qi = qkv[l][m * N + i];
                const double g = gy_ * qi;
                for (size_t j = 0; j < N; ++j) dw[l][i * N + j] = g;
                // S^l[i] = Σ_j w^l[i*N+j]
                double S = 0.0;
                for (size_t j = 0; j < N; ++j) S += last_w_[m][l][i * N + j];
                dq[l][m * N + i] = gy_ * S;
            }

        // conv backward: dw → dz, and (when the filter is generated) → dh_gen
        Tensor dz(L, P);
        dz.fill(0.0);
        for (size_t c = 0; c < P; ++c) {
            Tensor gws(1, L), vs(1, L), hs(1, L);
            for (size_t l = 0; l < L; ++l) {
                gws[0][l] = dw[l][c];
                vs[0][l] = z[l][c];
                hs[0][l] = h[l][c];
            }
            Tensor dzs = causal_conv_grad_v(gws, hs);
            for (size_t l = 0; l < L; ++l) dz[l][c] = dzs[0][l];
        }

        // outer-product backward: z[l][i*N+j] = k[l][i] * v[l][j], so
        //   dk[l][i] = Σ_j dz[l][i*N+j] * v[l][j]
        //   dv[l][j] = Σ_i dz[l][i*N+j] * k[l][i]
        // The two use DIFFERENT free indices: dk is scattered over i (the outer
        // loop), dv over j (the inner one). Collapsing dv onto i — as an
        // i-only loop with a j-sum does — silently puts the gradient on the
        // diagonal instead of scattering it, which FD catches as a structural
        // error with no clean factor (the error pattern depends on k, so no
        // single ratio).
        for (size_t l = 0; l < L; ++l)
            for (size_t i = 0; i < N; ++i) {
                double dki = 0.0;
                for (size_t j = 0; j < N; ++j)
                    dki += dz[l][i * N + j] * qkv[2 * L + l][m * N + j];
                dk[l][m * N + i] += dki;
            }
        for (size_t l = 0; l < L; ++l)
            for (size_t j = 0; j < N; ++j) {
                double dvj = 0.0;
                for (size_t i = 0; i < N; ++i)
                    dvj += dz[l][i * N + j] * qkv[L + l][m * N + i];
                dv[l][m * N + j] += dvj;
            }

        // Filter gradient. h[l][c] = h_gen[l][0] for every c, so
        //   dL/d h_gen[l] = Σ_c Σ_{t>=l} dw[t][c] * z[t-l][c]
        // When the filter is pinned by a test it is a constant and gets none.
        if (!pinned_[m]) {
            Tensor dgen(L, 1);
            dgen.fill(0.0);
            for (size_t c = 0; c < P; ++c) {
                Tensor gws(1, L), vs(1, L);
                for (size_t l = 0; l < L; ++l) { gws[0][l] = dw[l][c]; vs[0][l] = z[l][c]; }
                Tensor gks = causal_conv_grad_k(gws, vs);
                for (size_t l = 0; l < L; ++l) dgen[l][0] += gks[0][l];
            }
            Tensor zero_dskip(1, 1);
            zero_dskip.fill(0.0);
            filters_[m]->backward(dgen, zero_dskip);
        }
    }

    // in_proj backward.
    //
    // Shape contract (see Dense::backward): it computes
    //   grad_w = grad_output.transpose() * last_input
    // so grad_output must be (batch, out_features) = (L, 3*d_model) for a
    // Dense(d_model -> 3*d_model) whose last_input is (L, d_model). Passing the
    // (L, d_model) q/k/v slices separately would make Dense produce a
    // (d_model, d_model) grad that cannot accumulate into the (3*d_model,
    // d_model) grad_weights — the exact "Tensor dimensions mismatch" a shape
    // contract violation causes. So pass the full-width stack.
    Tensor gproj(L, 3 * d_model_);
    gproj.fill(0.0);
    for (size_t l = 0; l < L; ++l)
        for (size_t d = 0; d < d_model_; ++d) {
            gproj[l][d] = dq[l][d];
            gproj[l][d_model_ + d] = dk[l][d];
            gproj[l][2 * d_model_ + d] = dv[l][d];
        }
    last_input_grad_ = in_proj_.backward(gproj, 0.0);
    return last_input_grad_;
}

void MultiHeadHyenaOperator::zero_grad() {
    in_proj_.zero_grad();
    out_proj_.zero_grad();
    for (auto& f : filters_) f->zero_grad();
}

std::vector<Tensor*> MultiHeadHyenaOperator::parameters() {
    std::vector<Tensor*> p = {&in_proj_.weights, &in_proj_.bias,
                              &out_proj_.weights, &out_proj_.bias};
    for (auto& f : filters_) {
        auto fp = f->parameters();
        p.insert(p.end(), fp.begin(), fp.end());
    }
    return p;
}

std::vector<Tensor*> MultiHeadHyenaOperator::gradients() {
    std::vector<Tensor*> g = {&in_proj_.grad_weights, &in_proj_.grad_bias,
                              &out_proj_.grad_weights, &out_proj_.grad_bias};
    for (auto& f : filters_) {
        auto fg = f->gradients();
        g.insert(g.end(), fg.begin(), fg.end());
    }
    return g;
}

void MultiHeadHyenaOperator::update_weights(double lr) {
    in_proj_.update_weights(lr);
    out_proj_.update_weights(lr);
    // HyenaFilter has no update_weights(); the shipped HyenaOperator does the
    // same manual SGD over parameters()/gradients(). Mirror that.
    for (auto& f : filters_) {
        auto hp = f->parameters();
        auto hg = f->gradients();
        for (size_t i = 0; i < hp.size(); ++i)
            for (size_t j = 0; j < hp[i]->data.size(); ++j)
                hp[i]->data[j] -= lr * hg[i]->data[j];
    }
}

// ============================================================================
// MultiHeadHyenaBlock
// ============================================================================
MultiHeadHyenaBlock::MultiHeadHyenaBlock(size_t d_model, size_t seq_len,
                                         size_t num_heads, size_t ffn_mult)
    : d_model_(d_model), seq_len_(seq_len), num_heads_(num_heads),
      ffn_mult_(ffn_mult),
      operator_(d_model, seq_len, num_heads),
      ln1_(std::make_unique<RMSNorm>(d_model)),
      ln2_(std::make_unique<RMSNorm>(d_model))
{
    if (d_model == 0)
        throw std::invalid_argument("MultiHeadHyenaBlock: d_model must be > 0");
    if (seq_len == 0)
        throw std::invalid_argument("MultiHeadHyenaBlock: seq_len must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("MultiHeadHyenaBlock: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument(
            "MultiHeadHyenaBlock: d_model must be divisible by num_heads");
    if (ffn_mult > 0) {
        ffn_ = std::make_unique<SwiGLU<>>(d_model, ffn_mult * d_model);
        ffn_down_ = std::make_unique<Dense>(ffn_mult * d_model, d_model);
    }
}

Tensor MultiHeadHyenaBlock::forward(const Tensor& input) {
    if (input.cols != d_model_ || input.rows != seq_len_)
        throw std::invalid_argument("MultiHeadHyenaBlock: input must be (seq_len, d_model)");
    last_input_ = input;

    Tensor n1 = ln1_->forward(input);
    last_n1_ = n1;
    Tensor mix = operator_.forward(n1);
    last_mix_ = mix;

    Tensor u(input.rows, d_model_);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j) u[i][j] = input[i][j] + mix[i][j];
    last_u_ = u;

    if (!ffn_) return u;

    Tensor n2 = ln2_->forward(u);
    last_n2_ = n2;
    Tensor f = ffn_->forward(n2);
    Tensor fdown = ffn_down_->forward(f);

    Tensor y(input.rows, d_model_);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < d_model_; ++j) y[i][j] = u[i][j] + fdown[i][j];
    return y;
}

Tensor MultiHeadHyenaBlock::backward(const Tensor& grad_output, double /*lr*/) {
    Tensor dy = grad_output;

    if (ffn_) {
        Tensor dfdown = dy;
        Tensor df = ffn_down_->backward(dfdown, 0.0);
        Tensor dn2 = ffn_->backward(df, 0.0);
        Tensor du_norm = ln2_->backward(dn2, 0.0);
        for (size_t i = 0; i < du_norm.rows; ++i)
            for (size_t j = 0; j < du_norm.cols; ++j) dy[i][j] += du_norm[i][j];
    }

    Tensor dmix = dy;
    Tensor dn1 = operator_.backward(dmix, 0.0);
    // The residual add is u = input + mix(LN1(input)), so the input receives
    // the IDENTITY path as well as the mixer path:
    //   dL/dinput = dy + ln1.backward(dn1)
    // Returning only the ln1 term drops the identity contribution, which FD
    // sees as the block contributing nothing (numerically = 1, the pure
    // identity residual) while the analytical reads ~0.
    Tensor dn1_total = ln1_->backward(dn1, 0.0);
    last_input_grad_ = Tensor(seq_len_, d_model_);
    last_input_grad_.fill(0.0);
    for (size_t i = 0; i < seq_len_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            last_input_grad_[i][j] = dy[i][j] + dn1_total[i][j];
    return last_input_grad_;
}

void MultiHeadHyenaBlock::zero_grad() {
    operator_.zero_grad();
    ln1_->zero_grad();
    ln2_->zero_grad();
    if (ffn_) { ffn_->zero_grad(); ffn_down_->zero_grad(); }
}

std::vector<Tensor*> MultiHeadHyenaBlock::parameters() {
    std::vector<Tensor*> p = operator_.parameters();
    auto a = ln1_->parameters(); p.insert(p.end(), a.begin(), a.end());
    a = ln2_->parameters();        p.insert(p.end(), a.begin(), a.end());
    if (ffn_) {
        a = ffn_->parameters();    p.insert(p.end(), a.begin(), a.end());
        a = ffn_down_->parameters(); p.insert(p.end(), a.begin(), a.end());
    }
    return p;
}

std::vector<Tensor*> MultiHeadHyenaBlock::gradients() {
    std::vector<Tensor*> g = operator_.gradients();
    auto a = ln1_->gradients(); g.insert(g.end(), a.begin(), a.end());
    a = ln2_->gradients();        g.insert(g.end(), a.begin(), a.end());
    if (ffn_) {
        a = ffn_->gradients();    g.insert(g.end(), a.begin(), a.end());
        a = ffn_down_->gradients(); g.insert(g.end(), a.begin(), a.end());
    }
    return g;
}

void MultiHeadHyenaBlock::update_weights(double lr) {
    operator_.update_weights(lr);
    ln1_->update_weights(lr);
    ln2_->update_weights(lr);
    if (ffn_) { ffn_->update_weights(lr); ffn_down_->update_weights(lr); }
}

// ============================================================================
// MultiHeadHyenaModel
// ============================================================================
MultiHeadHyenaModel::MultiHeadHyenaModel(size_t input_dim, size_t d_model,
                                         size_t num_layers, size_t output_dim,
                                         size_t seq_len, size_t num_heads,
                                         size_t ffn_mult)
    // Member init order must match DECLARATION order in the header:
    // in_proj_, classifier_, then the size_t config fields. Putting the sizes
    // first triggers -Wreorder.
    : in_proj_(input_dim, d_model),
      classifier_(d_model, output_dim),
      d_model_(d_model), num_layers_(num_layers), output_dim_(output_dim),
      seq_len_(seq_len), num_heads_(num_heads), ffn_mult_(ffn_mult)
{
    if (input_dim == 0)
        throw std::invalid_argument("MultiHeadHyenaModel: input_dim must be > 0");
    if (d_model == 0)
        throw std::invalid_argument("MultiHeadHyenaModel: d_model must be > 0");
    if (num_layers == 0)
        throw std::invalid_argument("MultiHeadHyenaModel: num_layers must be > 0");
    if (output_dim == 0)
        throw std::invalid_argument("MultiHeadHyenaModel: output_dim must be > 0");
    if (seq_len == 0)
        throw std::invalid_argument("MultiHeadHyenaModel: seq_len must be > 0");
    if (num_heads == 0)
        throw std::invalid_argument("MultiHeadHyenaModel: num_heads must be > 0");
    if (d_model % num_heads != 0)
        throw std::invalid_argument(
            "MultiHeadHyenaModel: d_model must be divisible by num_heads");

    final_norm_ = std::make_unique<RMSNorm>(d_model);
    blocks.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i)
        blocks.emplace_back(d_model, seq_len, num_heads, ffn_mult);
}

Tensor MultiHeadHyenaModel::forward(const Tensor& input) {
    if (input.cols != in_proj_.weights.cols)
        throw std::invalid_argument("MultiHeadHyenaModel: input must be (seq_len, input_dim)");
    if (input.rows != seq_len_)
        throw std::invalid_argument("MultiHeadHyenaModel: input rows must equal seq_len");
    last_input_ = input;

    Tensor h = in_proj_.forward(input);
    last_proj_ = h;
    for (auto& b : blocks) h = b.forward(h);
    h = final_norm_->forward(h);

    Tensor pooled(1, d_model_);
    pooled.fill(0.0);
    for (size_t l = 0; l < seq_len_; ++l)
        for (size_t d = 0; d < d_model_; ++d) pooled[0][d] += h[l][d] / double(seq_len_);
    last_pooled_ = pooled;

    return classifier_.forward(pooled);
}

Tensor MultiHeadHyenaModel::backward(const Tensor& grad_output, double /*lr*/) {
    Tensor dpooled = classifier_.backward(grad_output, 0.0);

    Tensor dh(seq_len_, d_model_);
    dh.fill(0.0);
    for (size_t l = 0; l < seq_len_; ++l)
        for (size_t d = 0; d < d_model_; ++d) dh[l][d] = dpooled[0][d] / double(seq_len_);

    dh = final_norm_->backward(dh, 0.0);
    for (int i = (int)blocks.size() - 1; i >= 0; --i) dh = blocks[i].backward(dh, 0.0);

    Tensor ginput = in_proj_.backward(dh, 0.0);
    return ginput;
}

void MultiHeadHyenaModel::zero_grad() {
    in_proj_.zero_grad();
    final_norm_->zero_grad();
    classifier_.zero_grad();
    for (auto& b : blocks) b.zero_grad();
}

std::vector<Tensor*> MultiHeadHyenaModel::parameters() {
    std::vector<Tensor*> p = {&in_proj_.weights, &in_proj_.bias,
                              &classifier_.weights, &classifier_.bias};
    auto a = final_norm_->parameters(); p.insert(p.end(), a.begin(), a.end());
    for (auto& b : blocks) {
        a = b.parameters();
        p.insert(p.end(), a.begin(), a.end());
    }
    return p;
}

std::vector<Tensor*> MultiHeadHyenaModel::gradients() {
    std::vector<Tensor*> g = {&in_proj_.grad_weights, &in_proj_.grad_bias,
                              &classifier_.grad_weights, &classifier_.grad_bias};
    auto a = final_norm_->gradients(); g.insert(g.end(), a.begin(), a.end());
    for (auto& b : blocks) {
        a = b.gradients();
        g.insert(g.end(), a.begin(), a.end());
    }
    return g;
}

void MultiHeadHyenaModel::update_weights(double lr) {
    in_proj_.update_weights(lr);
    final_norm_->update_weights(lr);
    classifier_.update_weights(lr);
    for (auto& b : blocks) b.update_weights(lr);
}