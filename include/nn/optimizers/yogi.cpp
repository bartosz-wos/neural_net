#include "yogi.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction + validation
// ---------------------------------------------------------------------------
void Yogi::validate(double b1, double b2, double eps) {
    if (!(b1 > 0.0 && b1 < 1.0)) {
        throw std::invalid_argument("Yogi: beta1 must lie in (0, 1)");
    }
    if (!(b2 > 0.0 && b2 < 1.0)) {
        throw std::invalid_argument("Yogi: beta2 must lie in (0, 1)");
    }
    if (!(eps > 0.0)) {
        throw std::invalid_argument("Yogi: epsilon must be > 0");
    }
}

Yogi::Yogi(double lr_, double beta1_, double beta2_, double eps, double wd)
    : lr(lr_), beta1(beta1_), beta2(beta2_),
      epsilon(eps), weight_decay(wd), t(1)
{
    validate(beta1, beta2, epsilon);
}

void Yogi::set_lr(double new_lr) { lr = new_lr; }
void Yogi::set_beta1(double new_b1) { validate(new_b1, beta2, epsilon); beta1 = new_b1; }
void Yogi::set_beta2(double new_b2) { validate(beta1, new_b2, epsilon); beta2 = new_b2; }
void Yogi::set_epsilon(double new_eps) { validate(beta1, beta2, new_eps); epsilon = new_eps; }
void Yogi::set_weight_decay(double new_wd) { weight_decay = new_wd; }

// ---------------------------------------------------------------------------
// State plumbing
// ---------------------------------------------------------------------------
void Yogi::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<YogiState> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        YogiState st;
        st.m = Tensor(p.rows, p.cols);
        st.m.fill(0.0);
        st.v = Tensor(p.rows, p.cols);
        st.v.fill(0.0);
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

Tensor Yogi::get_m(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) {
        return Tensor(0, 0);
    }
    return it->second[param_idx].m;
}

// Returns a copy of the v tensor for (layer, param_idx), or (0, 0) if absent.
Tensor Yogi::get_v(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) {
        return Tensor(0, 0);
    }
    return it->second[param_idx].v;
}

// ---------------------------------------------------------------------------
// Per-parameter update (the Yogi "soul")
// ---------------------------------------------------------------------------
void Yogi::update_param(Tensor* param, Tensor* grad, YogiState& st,
                         double lr_, double b1, double b2,
                         double eps, double wd, double b1_c, double b2_c) const
{
    const double one_minus_b2 = 1.0 - b2;
    for (size_t i = 0; i < grad->rows; ++i) {
        for (size_t j = 0; j < grad->cols; ++j) {
            double g = (*grad)[i][j];

            // --- First-moment EMA (same as Adam) ---
            double m_prev = st.m[i][j];
            double m_new  = b1 * m_prev + (1.0 - b1) * g;
            st.m[i][j]    = m_new;

            // --- Yogi rule for v_t ---
            // v_t = v_{t-1} − (1−β2) · sign(v_{t-1} − g²) · g²
            //
            // Properties:
            //   * Always strictly ≥ 0 if v_{t-1} ≥ 0 (the sign=-1 branch adds
            //     (1−β2)·g², the sign=+1 branch subtracts (1−β2)·g²).
            //   * If we set v_prev = 0, the rule reduces to
            //     v_new = (1−β2)·g² — same value as Adam at v_prev=0.
            //   * "delta = 0" (theoretically zero-measure) is treated as a
            //     no-op (v_new = v_prev), matching the paper's pseudocode.
            double g2   = g * g;
            double v_prev = st.v[i][j];
            double delta = v_prev - g2;
            double v_new;
            if      (delta > 0.0) v_new = v_prev - one_minus_b2 * g2;  // sign = +1
            else if (delta < 0.0) v_new = v_prev + one_minus_b2 * g2;  // sign = -1
            else                  v_new = v_prev;                       // delta = 0, no-op

            // The update above can take v_new slightly negative when delta>0
            // and (1−β2)·g² > v_prev (rare but possible at t=1 with v_prev ≈ 0).
            // No-op at zero, but to avoid even theoretical negatives we keep a
            // floor at 0. The paper allows v_t to go negative in the algorithm;
            // we additionally use |v̂_t| (sqrt of absolute value) in the denom
            // so the sign issue doesn't affect the step.
            if (v_new < 0.0) v_new = 0.0;
            st.v[i][j] = v_new;

            // --- Bias-corrected moments ---
            double m_hat = m_new / b1_c;
            double v_hat = v_new / b2_c;

            // --- Denom + step (|v̂_t| to handle transient v<0 robustly) ---
            double denom = std::sqrt(std::abs(v_hat)) + eps;
            double step  = m_hat / denom;

            // --- Decoupled weight decay (AdamW style) ---
            if (wd > 0.0) {
                (*param)[i][j] -= lr_ * wd * (*param)[i][j];
            }

            // --- Final parameter update ---
            (*param)[i][j] -= lr_ * step;
        }
    }
}

// ---------------------------------------------------------------------------
// Step
// ---------------------------------------------------------------------------
void Yogi::step(Model& model) {
    // Bias-correction terms for this timestep.
    // At t=1: b1_c = 1 - b1, b2_c = 1 - b2  (the paper convention).
    double b1_c = 1.0 - std::pow(beta1, t);
    double b2_c = 1.0 - std::pow(beta2, t);

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer       = model.layers[li];
        auto* layer_ptr   = layer.get();
        auto params       = layer->parameters();
        auto grads        = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& state_vec   = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i],
                         state_vec[i],
                         lr, beta1, beta2,
                         epsilon, weight_decay,
                         b1_c, b2_c);
        }

        layer->zero_grad();
    }

    ++t;
}
