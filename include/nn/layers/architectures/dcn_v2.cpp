#include "nn/layers/architectures/dcn_v2.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace {

// File-local matmul: (M, K) × (K, N) → (M, N).
// Tensor has no built-in matmul — this is the canonical implementation.
Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows)
        throw std::invalid_argument("matmul: inner dim mismatch");
    Tensor C(A.rows, B.cols);
    C.fill(0.0);
    for (size_t i = 0; i < A.rows; ++i)
        for (size_t k = 0; k < A.cols; ++k) {
            double a = A(i, k);
            for (size_t j = 0; j < B.cols; ++j)
                C(i, j) += a * B(k, j);
        }
    return C;
}

// Elementwise ReLU forward.
Tensor relu_forward(const Tensor& x) {
    Tensor y(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j) {
            double v = x(i, j);
            y(i, j) = (v > 0.0) ? v : 0.0;
        }
    return y;
}

} // anonymous namespace


// ============================================================================
// CrossLayer
// ============================================================================
CrossLayer::CrossLayer(size_t d_, size_t r_)
    : d(d_), r(r_),
      C(Tensor::random(r_, d_, 0.1)),
      U(Tensor::random(d_, r_, 0.1)),
      gC(Tensor::zeros(r_, d_)),
      gU(Tensor::zeros(d_, r_)),
      last_p(), last_q(), last_x_lm1() {
    if (d_ == 0) throw std::invalid_argument("CrossLayer: d must be > 0");
    if (r_ == 0) throw std::invalid_argument("CrossLayer: r must be > 0");
    if (r_ > d_) throw std::invalid_argument("CrossLayer: r must be <= d");
}

void CrossLayer::zero_grad() {
    gC.fill(0.0);
    gU.fill(0.0);
}

std::vector<Tensor*> CrossLayer::parameters() {
    return {&C, &U};
}

std::vector<Tensor*> CrossLayer::gradients() {
    return {&gC, &gU};
}


// ============================================================================
// CrossNetwork
// ============================================================================
CrossNetwork::CrossNetwork(size_t d, size_t num_layers, size_t r)
    : d_(d), num_layers_(num_layers), r_(r),
      last_x0_(0, 0), last_xL_(0, 0) {
    if (d == 0) throw std::invalid_argument("CrossNetwork: d must be > 0");
    if (num_layers == 0) throw std::invalid_argument("CrossNetwork: num_layers must be > 0");
    if (r == 0) throw std::invalid_argument("CrossNetwork: r must be > 0");
    if (r > d) throw std::invalid_argument("CrossNetwork: r must be <= d");
    layers_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.emplace_back(d, r);
    }
}

Tensor CrossNetwork::forward(const Tensor& input) {
    if (input.cols != d_)
        throw std::invalid_argument("CrossNetwork::forward: input cols mismatch");
    last_x0_ = input.clone();
    Tensor x = input.clone();  // x_l, starting at x_0
    for (auto& L : layers_) {
        L.last_x_lm1 = x.clone();
        // p_l = C · x_{l-1}      (B, r) = (B, d) × (d, r)
        // C is (r, d); use C.T() (d, r) to get (B, d) × (d, r)
        L.last_p = matmul(x, L.C.transpose());
        // q_l = U · p_l          (B, d) = (B, r) × (r, d)
        // U is (d, r); use U.T() (r, d) to get (B, r) × (r, d)
        L.last_q = matmul(L.last_p, L.U.transpose());
        // x_l = x_0 ⊙ q_l + x_{l-1}
        Tensor x_new(input.rows, input.cols);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                x_new(i, j) = last_x0_(i, j) * L.last_q(i, j) + x(i, j);
        x = x_new;
    }
    last_xL_ = x.clone();
    return x;
}

Tensor CrossNetwork::backward(const Tensor& grad_output, double /*lr*/) {
    const size_t B = last_x0_.rows;
    const size_t d = d_;
    Tensor dx0(B, d);
    dx0.fill(0.0);
    Tensor dx_l = grad_output.clone();   // (B, d)

    for (size_t idx = num_layers_; idx > 0; --idx) {
        auto& L = layers_[idx - 1];

        // dq = x_0 ⊙ dx_l       (B, d)
        Tensor dq(B, d);
        for (size_t i = 0; i < B; ++i)
            for (size_t j = 0; j < d; ++j)
                dq(i, j) = last_x0_(i, j) * dx_l(i, j);

        // dU_l = dq^T · p_l    (d, r)
        // dU(i, j) = Σ_b dq(b, i) * p_l(b, j)
        Tensor dU(d, L.r);
        dU.fill(0.0);
        for (size_t b = 0; b < B; ++b)
            for (size_t i = 0; i < d; ++i)
                for (size_t j = 0; j < L.r; ++j)
                    dU(i, j) += dq(b, i) * L.last_p(b, j);

        // dC = (U^T · dq)^T · x_{l-1}      (r, d)
        // For each batch b: tmp_b(r, j) = Σ_i U_l(i, r) * dq(b, j)
        // Then dC(r, j) = Σ_b tmp_b(r, j) * x_{l-1}(b, j)
        Tensor dC(L.r, d);
        dC.fill(0.0);
        for (size_t b = 0; b < B; ++b) {
            // tmp_b(r, j) = Σ_i U(i, r) * dq(b, j)   ; since dq(b, j) doesn't depend on i,
            //                                                     this is dq(b, j) * (Σ_i U(i, r)).
            // But that's WRONG — let me re-derive:
            // q[b, j] = Σ_r U(j, r) · (Σ_k C(r, k) · x[b, k])
            // dq[b, j] = ∂L/∂q[b, j] = x_0[b, j] · dx_l[b, j]
            // x_l[b, j] = x_0[b, j] · q[b, j] + x_{l-1}[b, j]
            // ∂x_l[b, j] / ∂C(r, k) = x_0[b, j] · ∂q[b, j] / ∂C(r, k)
            //                        = x_0[b, j] · U(j, r) · x_{l-1}[b, k]
            // dC(r, k) = Σ_{b, j} dx_l[b, j] · x_0[b, j] · U(j, r) · x_{l-1}[b, k]
            //          = Σ_b x_{l-1}[b, k] · Σ_j U(j, r) · dq[b, j]   (since dq = x_0 ⊙ dx_l)
            //          = Σ_b x_{l-1}[b, k] · tmp_b(r)
            // where tmp_b(r) = Σ_j U(j, r) · dq[b, j]
            for (size_t k = 0; k < d; ++k) {
                for (size_t r = 0; r < L.r; ++r) {
                    double tmp_r = 0;
                    for (size_t j = 0; j < d; ++j)
                        tmp_r += L.U(j, r) * dq(b, j);
                    dC(r, k) += tmp_r * L.last_x_lm1(b, k);
                }
            }
        }

        // d x_{l-1} = d x_l + (U · C)^T · dq       (B, d)
        // For each batch b:
        //   for j: dx_prev(b, j) = dx_l(b, j) + Σ_{r, i} U(i, r) * C(r, i) * dq(b, j)
        //        = dx_l(b, j) + dq(b, j) * Σ_{r, i} U(i, r) * C(r, i)
        // The inner sum Σ_{r, i} U(i, r) * C(r, i) is just the trace of M = U · C
        // but i, r are decoupled — actually it's the (j, j) diagonal of M · M^T... no.
        // Re-derive: (U · C)^T = C^T · U^T, shape (d, d).
        // ((U · C)^T · dq)(b, j) = Σ_k (U·C)^T(j, k) * dq(b, k)
        //                        = Σ_k Σ_r C^T(j, r) · U^T(r, k) · dq(b, k)
        //                        = Σ_r C^T(j, r) · Σ_k U^T(r, k) · dq(b, k)
        //                        = Σ_r C(r, j) · (U^T · dq)(b, r)   where U^T(r, k) = U(k, r)
        // So: dx_prev(b, j) = dx_l(b, j) + Σ_r C(r, j) * Σ_k U(k, r) * dq(b, k)
        //                    = dx_l(b, j) + Σ_r C(r, j) * Ut_dq(b, r)
        // where Ut_dq(b, r) = Σ_k U(k, r) * dq(b, k)
        Tensor Ut_dq(B, L.r);
        Ut_dq.fill(0.0);
        for (size_t b = 0; b < B; ++b)
            for (size_t r = 0; r < L.r; ++r)
                for (size_t j = 0; j < d; ++j)         // <-- was `k`, should be `j`
                    Ut_dq(b, r) += L.U(j, r) * dq(b, j);   // <-- U(j, r) not U(k, r); dq(b, j) not dq(b, k)

        Tensor dx_prev(B, d);
        for (size_t b = 0; b < B; ++b)
            for (size_t j = 0; j < d; ++j) {
                double s = dx_l(b, j);
                for (size_t r = 0; r < L.r; ++r)
                    s += L.C(r, j) * Ut_dq(b, r);
                dx_prev(b, j) = s;
            }

        // Accumulate d x_0 contribution from the cross term's elementwise part:
        // every layer contributes dx_l ⊙ q_l (from the δ_{j,k} · q_l[b, j] term
        // in ∂(x_0 ⊙ q_l)[b, j] / ∂x_0[b, k]).
        for (size_t i = 0; i < B; ++i)
            for (size_t j = 0; j < d; ++j)
                dx0(i, j) += dx_l(i, j) * L.last_q(i, j);

        // For layer 1 (idx == 1 in backward order), x_{l-1} = x_0 — so we get two
        // additional contributions to dx_0:
        //   1. The residual contribution dx_l itself (from x_l = x_0 ⊙ q_l + x_0).
        //   2. The cross-chain contribution dq · (U·C) (from q_1 depending on x_0).
        //      This is exactly (dx_prev - dx_l) since dx_prev = dx_l + (U·C)^T · dq
        //      is the gradient to x_0 through the chain.
        if (idx == 1) {
            for (size_t i = 0; i < B; ++i)
                for (size_t j = 0; j < d; ++j) {
                    dx0(i, j) += dx_l(i, j);            // residual contribution
                    dx0(i, j) += dx_prev(i, j) - dx_l(i, j);  // cross-chain contribution
                }
        }

        // Accumulate gradients into gU, gC
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < L.r; ++j)
                L.gU(i, j) += dU(i, j);
        for (size_t i = 0; i < L.r; ++i)
            for (size_t j = 0; j < d; ++j)
                L.gC(i, j) += dC(i, j);

        dx_l = dx_prev;
    }

    return dx0;
}

void CrossNetwork::update_weights(double lr) {
    for (auto& L : layers_) {
        for (size_t i = 0; i < L.d; ++i)
            for (size_t j = 0; j < L.r; ++j)
                L.U(i, j) -= lr * L.gU(i, j);
        for (size_t i = 0; i < L.r; ++i)
            for (size_t j = 0; j < L.d; ++j)
                L.C(i, j) -= lr * L.gC(i, j);
    }
}

void CrossNetwork::zero_grad() {
    for (auto& L : layers_) L.zero_grad();
}

std::vector<Tensor*> CrossNetwork::parameters() {
    std::vector<Tensor*> out;
    out.reserve(2 * num_layers_);
    for (auto& L : layers_) {
        auto p = L.parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

std::vector<Tensor*> CrossNetwork::gradients() {
    std::vector<Tensor*> out;
    out.reserve(2 * num_layers_);
    for (auto& L : layers_) {
        auto g = L.gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    return out;
}

Tensor CrossNetwork::get_weights() const {
    return layers_.empty() ? Tensor() : layers_[0].C;
}

Tensor CrossNetwork::get_gradients() const {
    return layers_.empty() ? Tensor() : layers_[0].gC;
}


// ============================================================================
// DCNv2
// ============================================================================
DCNv2::DCNv2(size_t input_dim, size_t cross_layers, size_t cross_r,
             const std::vector<size_t>& deep_dims, size_t output_dim)
    : input_dim_(input_dim), output_dim_(output_dim),
      deep_dims_(deep_dims),
      cross_(input_dim, cross_layers, cross_r),
      stack_(0, 0),  // placeholder; real value set in body after validation
      last_xL_(0, 0), last_h_deep_(0, 0), last_concat_(0, 0) {
    if (input_dim == 0) throw std::invalid_argument("DCNv2: input_dim must be > 0");
    if (output_dim == 0) throw std::invalid_argument("DCNv2: output_dim must be > 0");
    if (deep_dims.empty()) throw std::invalid_argument("DCNv2: deep_dims must be non-empty");

    // Build deep tower: input_dim → deep_dims[0] → deep_dims[1] → ... → deep_dims[-1]
    size_t prev = input_dim;
    for (size_t dd : deep_dims) {
        if (dd == 0) throw std::invalid_argument("DCNv2: deep_dims must all be > 0");
        deep_.emplace_back(std::make_unique<Dense>(prev, dd));
        prev = dd;
    }
    // Stack Dense: [x_L (d=input_dim) ; h_deep (deep_dims[-1])] → output_dim
    stack_ = Dense(input_dim + deep_dims.back(), output_dim);

    // Pre-allocate per-deep-layer pre-ReLU caches
    last_h_pre_relu_.resize(deep_dims.size());
}

Tensor DCNv2::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("DCNv2::forward: input cols mismatch");

    // Cross tower
    last_xL_ = cross_.forward(input);   // (B, input_dim)

    // Deep tower
    Tensor h = input;
    for (size_t i = 0; i < deep_.size(); ++i) {
        h = deep_[i]->forward(h);
        last_h_pre_relu_[i] = h.clone();   // cache pre-ReLU
        h = relu_forward(h);
    }
    last_h_deep_ = h;

    // Concatenate cross output and deep output, then stack Dense
    last_concat_ = last_xL_.concatenate(h, /*along_cols=*/true);
    return stack_.forward(last_concat_);
}

Tensor DCNv2::backward(const Tensor& grad_output, double /*lr*/) {
    // 1. Backward through stack layer
    Tensor d_concat = stack_.backward(grad_output, 0.0);

    // 2. Split into d_xL (first input_dim cols) and d_h (remaining cols)
    const size_t d = input_dim_;
    const size_t last_deep_cols = deep_.back()->weights.rows;
    if (d_concat.cols != d + last_deep_cols)
        throw std::logic_error("DCNv2::backward: concat column count mismatch");

    Tensor d_xL(d_concat.rows, d);
    Tensor d_h(d_concat.rows, last_deep_cols);
    for (size_t i = 0; i < d_concat.rows; ++i) {
        for (size_t j = 0; j < d; ++j)
            d_xL(i, j) = d_concat(i, j);
        for (size_t j = 0; j < last_deep_cols; ++j)
            d_h(i, j) = d_concat(i, d + j);
    }

    // 3. Backward through deep tower in REVERSE order.
    // For each layer i (going from last to first):
    //   Apply ReLU mask: d_h *= (last_h_pre_relu_[i] > 0)
    //   d_h = deep_[i]->backward(d_h)
    //   d_h now has the gradient to the input of deep_[i]
    Tensor d_x0_from_deep;
    for (size_t ii = deep_.size(); ii > 0; --ii) {
        const size_t i = ii - 1;
        // ReLU mask
        for (size_t b = 0; b < d_h.rows; ++b)
            for (size_t j = 0; j < d_h.cols; ++j)
                if (last_h_pre_relu_[i](b, j) <= 0.0)
                    d_h(b, j) = 0.0;
        d_h = deep_[i]->backward(d_h, 0.0);
        if (i == 0) {
            d_x0_from_deep = d_h;
        }
    }

    // 4. Backward through cross network
    Tensor d_x0_from_cross = cross_.backward(d_xL, 0.0);

    // 5. Add the deep tower's input gradient to the cross tower's input gradient
    // Both are gradient w.r.t. the model input x.
    Tensor d_x0(d_x0_from_cross.rows, d_x0_from_cross.cols);
    for (size_t i = 0; i < d_x0.rows; ++i)
        for (size_t j = 0; j < d_x0.cols; ++j)
            d_x0(i, j) = d_x0_from_cross(i, j) + d_x0_from_deep(i, j);

    return d_x0;
}

void DCNv2::update_weights(double lr) {
    cross_.update_weights(lr);
    for (auto& d : deep_) d->update_weights(lr);
    stack_.update_weights(lr);
}

void DCNv2::zero_grad() {
    cross_.zero_grad();
    for (auto& d : deep_) d->zero_grad();
    stack_.zero_grad();
}

std::vector<Tensor*> DCNv2::parameters() {
    std::vector<Tensor*> out;
    auto cp = cross_.parameters();
    out.insert(out.end(), cp.begin(), cp.end());
    for (auto& d : deep_) {
        auto p = d->parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    auto sp = stack_.parameters();
    out.insert(out.end(), sp.begin(), sp.end());
    return out;
}

std::vector<Tensor*> DCNv2::gradients() {
    std::vector<Tensor*> out;
    auto cg = cross_.gradients();
    out.insert(out.end(), cg.begin(), cg.end());
    for (auto& d : deep_) {
        auto g = d->gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    auto sg = stack_.gradients();
    out.insert(out.end(), sg.begin(), sg.end());
    return out;
}

Tensor DCNv2::get_weights() const { return cross_.get_weights(); }
Tensor DCNv2::get_gradients() const { return cross_.get_gradients(); }

DCNv2 DCNv2::clone() const {
    // DCNv2 constructor sets up cross_, deep_, and stack_ with default random
    // init — we throw that away and replace with deep copies of the source's
    // tensors.  For FD tests this is fine: we only mutate parameters and then
    // call forward().
    DCNv2 out(input_dim_, cross_.num_layers(), cross_.r(), deep_dims_, output_dim_);
    out.cross_ = cross_;                // CrossNetwork is copyable (vector<Tensor> only)
    out.deep_.clear();
    for (const auto& d : deep_) {
        out.deep_.emplace_back(std::make_unique<Dense>(*d));   // Dense copyable
    }
    out.stack_ = stack_;                // Dense copyable
    out.last_xL_ = last_xL_;
    out.last_h_deep_ = last_h_deep_;
    out.last_h_pre_relu_ = last_h_pre_relu_;
    out.last_concat_ = last_concat_;
    return out;
}


// ============================================================================
// DCNv2Regression
// ============================================================================
DCNv2Regression::DCNv2Regression(size_t input_dim, size_t output_dim,
                                 size_t cross_r, size_t cross_layers,
                                 size_t deep_hidden, size_t deep_layers)
    : model_(input_dim,
             cross_layers,
             (cross_r == 0) ? std::max<size_t>(1, input_dim / 4) : cross_r,
             std::vector<size_t>(deep_layers, deep_hidden),
             output_dim) {
    if (deep_hidden == 0) throw std::invalid_argument("DCNv2Regression: deep_hidden must be > 0");
    if (deep_layers == 0) throw std::invalid_argument("DCNv2Regression: deep_layers must be > 0");
}

Tensor DCNv2Regression::forward(const Tensor& input)    { return model_.forward(input); }
Tensor DCNv2Regression::backward(const Tensor& g, double lr) { return model_.backward(g, lr); }
void  DCNv2Regression::update_weights(double lr)         { model_.update_weights(lr); }
void  DCNv2Regression::zero_grad()                      { model_.zero_grad(); }
std::vector<Tensor*> DCNv2Regression::parameters()      { return model_.parameters(); }
std::vector<Tensor*> DCNv2Regression::gradients()       { return model_.gradients(); }
Tensor DCNv2Regression::get_weights() const             { return model_.get_weights(); }
Tensor DCNv2Regression::get_gradients() const           { return model_.get_gradients(); }