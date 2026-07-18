#include "adan.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction + validation
// ---------------------------------------------------------------------------
void Adan::validate(double b1, double b2, double b3, double eps) {
    if (!(b1 >= 0.0 && b1 < 1.0)) {
        throw std::invalid_argument("Adan: beta1 must lie in [0, 1)");
    }
    if (!(b2 >= 0.0 && b2 < 1.0)) {
        throw std::invalid_argument("Adan: beta2 must lie in [0, 1)");
    }
    if (!(b3 >= 0.0 && b3 < 1.0)) {
        throw std::invalid_argument("Adan: beta3 must lie in [0, 1)");
    }
    if (!(eps > 0.0)) {
        throw std::invalid_argument("Adan: epsilon must be > 0");
    }
}

Adan::Adan(double lr_, double beta1_, double beta2_, double beta3_,
           double eps, double wd, bool no_prox_)
    : lr(lr_), beta1(beta1_), beta2(beta2_), beta3(beta3_),
      epsilon(eps), weight_decay(wd), no_prox(no_prox_), t(1)
{
    validate(beta1, beta2, beta3, epsilon);
}

void Adan::set_lr(double new_lr)            { lr = new_lr; }
void Adan::set_beta1(double new_b1)         { validate(new_b1, beta2, beta3, epsilon); beta1 = new_b1; }
void Adan::set_beta2(double new_b2)         { validate(beta1, new_b2, beta3, epsilon); beta2 = new_b2; }
void Adan::set_beta3(double new_b3)         { validate(beta1, beta2, new_b3, epsilon); beta3 = new_b3; }
void Adan::set_epsilon(double new_eps)      { validate(beta1, beta2, beta3, new_eps); epsilon = new_eps; }
void Adan::set_weight_decay(double new_wd)  { weight_decay = new_wd; }
void Adan::set_no_prox(bool new_no_prox)    { no_prox = new_no_prox; }

// ---------------------------------------------------------------------------
// State plumbing
// ---------------------------------------------------------------------------
void Adan::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<AdanState> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        AdanState st;
        st.m             = Tensor(p.rows, p.cols);
        st.m.fill(0.0);
        st.v             = Tensor(p.rows, p.cols);
        st.v.fill(0.0);
        st.n             = Tensor(p.rows, p.cols);
        st.n.fill(0.0);
        // neg_prev_grad is initialized to -g in step() (after the first
        // forward/backward), not here. We size it but leave values at 0
        // until the first step() populates it.
        st.neg_prev_grad = Tensor(p.rows, p.cols);
        st.neg_prev_grad.fill(0.0);
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

Tensor Adan::get_m(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) {
        return Tensor(0, 0);
    }
    return it->second[param_idx].m;
}

Tensor Adan::get_v(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) {
        return Tensor(0, 0);
    }
    return it->second[param_idx].v;
}

Tensor Adan::get_n(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end() || param_idx >= it->second.size()) {
        return Tensor(0, 0);
    }
    return it->second[param_idx].n;
}

// ---------------------------------------------------------------------------
// Per-parameter update (the Adan "soul")
// ---------------------------------------------------------------------------
// Matches sail-sg/Adan _single_tensor_adan line-by-line. The trick:
//   neg_grad_or_diff = neg_prev_grad + grad   (in-place add to neg_prev_grad)
//                     = -g_{k-1} + g_k       (gradient difference)
//   then reuse the same buffer for the n-step computation:
//                     neg_grad_or_diff = β2 * (g_k - g_{k-1}) + g_k
// We split into two stages to keep the math obvious.
void Adan::update_param(Tensor* param, Tensor* grad, AdanState& st,
                         double lr_, double b1, double b2, double b3,
                         double eps, double wd, bool   no_prox_,
                         double bc1, double bc2, double bc3_sqrt) const
{
    const double one_minus_b1 = 1.0 - b1;
    const double one_minus_b2 = 1.0 - b2;
    const double one_minus_b3 = 1.0 - b3;
    const double b2_over_bc2  = b2 / bc2;

    for (size_t i = 0; i < grad->rows; ++i) {
        for (size_t j = 0; j < grad->cols; ++j) {
            double g = (*grad)[i][j];

            // Recover g_{k-1} from the stored negated value, then update
            // neg_prev_grad in place to hold the gradient-difference g_k - g_{k-1}.
            //
            // On the very first step, neg_prev_grad holds -g (set in step()
            // BEFORE update_param is called), so the diff is 0 — matches
            // the sail-sg convention exactly.
            double neg_prev = st.neg_prev_grad[i][j];      // = -g_{k-1}
            double diff     = g + neg_prev;                // = g_k - g_{k-1}
            st.neg_prev_grad[i][j] = -g;                   // store -(g_k) for next step

            // --- First-moment EMA (m_t) ---
            double m_prev = st.m[i][j];
            double m_new  = b1 * m_prev + one_minus_b1 * g;
            st.m[i][j]    = m_new;

            // --- Gradient-difference EMA (v_t) ---
            // v_t = β2 * v_{t-1} + (1 - β2) * (g_k - g_{k-1})
            double v_prev = st.v[i][j];
            double v_new  = b2 * v_prev + one_minus_b2 * diff;
            st.v[i][j]    = v_new;

            // --- Third-moment EMA (n_t) ---
            // n_t = β3 * n_{t-1} + (1 - β3) * (g_k + β2 * (g_k - g_{k-1}))²
            // (Note: this is the OFFICIAL sail-sg convention; the paper's
            // Algorithm 1 has β3 and 1-β3 swapped. See adan.h header note.)
            double inside = g + b2 * diff;
            double n_prev = st.n[i][j];
            double n_new  = b3 * n_prev + one_minus_b3 * inside * inside;
            st.n[i][j]    = n_new;

            // --- Bias-corrected denom: sqrt(n_t / (1 - β3^t)) + ε ---
            // 1 - β3^t = bc3_sqrt², so n / bc3 = n / (bc3_sqrt²)
            double denom = std::sqrt(n_new / (bc3_sqrt * bc3_sqrt)) + eps;

            // --- Step components (matches sail-sg's `step_size` and `step_size_diff`) ---
            double step_size      = lr_ / bc1;
            double step_size_diff = lr_ * b2_over_bc2;
            double step_m  = step_size * m_new / denom;
            double step_v  = step_size_diff * v_new / denom;

            // --- Decoupled vs prox weight decay ---
            if (no_prox_) {
                // AdamW-style decoupled path
                (*param)[i][j] *= (1.0 - lr_ * wd);
                (*param)[i][j] -= step_m + step_v;
            } else {
                // Integrated prox path (paper default; sail-sg no_prox=False)
                (*param)[i][j] = ((*param)[i][j] - step_m - step_v) / (1.0 + lr_ * wd);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Step
// ---------------------------------------------------------------------------
void Adan::step(Model& model) {
    // Bias-correction terms for this timestep.
    // At t=1: bc1 = 1 - β1, bc2 = 1 - β2, bc3_sqrt = sqrt(1 - β3).
    double bc1       = 1.0 - std::pow(beta1, t);
    double bc2       = 1.0 - std::pow(beta2, t);
    double bc3_sqrt  = std::sqrt(1.0 - std::pow(beta3, t));

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer       = model.layers[li];
        auto* layer_ptr   = layer.get();
        auto params       = layer->parameters();
        auto grads        = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& state_vec   = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            // On the first step per param, initialize neg_prev_grad to -(grad)
            // so the diff (g - prev_g) = 0 — matches sail-sg's convention
            // where `neg_pre_grad` is initialized to `p.grad * -1` on step 1.
            // (Subsequent steps reuse the stored -(g_{k-1}) from the previous
            // update.)
            if (t == 1) {
                for (size_t r = 0; r < grads[i]->rows; ++r) {
                    for (size_t c = 0; c < grads[i]->cols; ++c) {
                        state_vec[i].neg_prev_grad[r][c] = -(*grads[i])[r][c];
                    }
                }
            }

            update_param(params[i], grads[i],
                         state_vec[i],
                         lr, beta1, beta2, beta3,
                         epsilon, weight_decay, no_prox,
                         bc1, bc2, bc3_sqrt);
        }

        layer->zero_grad();
    }

    ++t;
}