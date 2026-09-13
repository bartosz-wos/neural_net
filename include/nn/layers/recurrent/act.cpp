#include "act.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <algorithm>

ACTRecurrentLayer::ACTRecurrentLayer(int input_dim, int hidden_size, int max_steps,
                                     double ponder_penalty)
    : input_dim_(input_dim),
      hidden_size_(hidden_size),
      max_steps_(max_steps),
      ponder_penalty_(ponder_penalty),
      W_xh_(hidden_size, input_dim),
      W_hh_(hidden_size, hidden_size),
      b_(hidden_size, 1),
      grad_W_xh_(hidden_size, input_dim),
      grad_W_hh_(hidden_size, hidden_size),
      grad_b_(hidden_size, 1),
      W_halt_(hidden_size, 1),
      b_halt_(1, 1),
      grad_W_halt_(hidden_size, 1),
      grad_b_halt_(1, 1),
      halt_step_(max_steps - 1),
      total_ponder_(0.0)
{
    if (input_dim_ <= 0) throw std::invalid_argument("ACTRecurrentLayer: input_dim must be > 0");
    if (hidden_size_ <= 0) throw std::invalid_argument("ACTRecurrentLayer: hidden_size must be > 0");
    if (max_steps_ <= 0) throw std::invalid_argument("ACTRecurrentLayer: max_steps must be > 0");
    if (ponder_penalty_ < 0.0) throw std::invalid_argument("ACTRecurrentLayer: ponder_penalty must be >= 0");

    random_init(42);
    zero_grad();
}

void ACTRecurrentLayer::random_init(unsigned seed) {
    // Uniform init (similar to GRU): keep weights small so p_t does not saturate.
    double scale_xh = std::sqrt(6.0 / (input_dim_ + hidden_size_));
    double scale_hh = std::sqrt(6.0 / (hidden_size_ + hidden_size_));
    double scale_halt = std::sqrt(6.0 / (hidden_size_ + 1));

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis_xh(-scale_xh, scale_xh);
    std::uniform_real_distribution<> dis_hh(-scale_hh, scale_hh);
    std::uniform_real_distribution<> dis_halt(-scale_halt, scale_halt);

    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < input_dim_; ++j) W_xh_[i][j] = dis_xh(gen);
        for (int j = 0; j < hidden_size_; ++j) W_hh_[i][j] = dis_hh(gen);
        b_[i][0] = 0.0;
        // W_halt init: small but nonzero so the halting head has signal.
        for (int j = 0; j < 1; ++j) W_halt_[i][j] = dis_halt(gen);
    }
    b_halt_[0][0] = 0.0;
}

void ACTRecurrentLayer::zero_all_weights() {
    W_xh_.fill(0.0);
    W_hh_.fill(0.0);
    b_.fill(0.0);
    W_halt_.fill(0.0);
    b_halt_.fill(0.0);
}

double ACTRecurrentLayer::ponder_cost() const {
    return ponder_penalty_ * total_ponder_;
}

Tensor ACTRecurrentLayer::forward(const Tensor& input) {
    if (input.rows != 1) {
        throw std::invalid_argument("ACTRecurrentLayer: v1 only supports batch size 1");
    }
    if (input.cols != static_cast<size_t>(max_steps_ * input_dim_)) {
        throw std::invalid_argument(
            "ACTRecurrentLayer: input.cols must equal max_steps * input_dim");
    }

    // 1. Cache inputs.
    inputs_ = Tensor(1, max_steps_ * input_dim_);
    for (size_t j = 0; j < inputs_.cols; ++j) inputs_[0][j] = input[0][j];

    // 2. Allocate per-step caches.
    hidden_states_ = Tensor(max_steps_ + 1, hidden_size_);
    outputs_       = Tensor(max_steps_, hidden_size_);
    halt_probs_    = Tensor(1, max_steps_);
    halt_logits_   = Tensor(1, max_steps_);
    weights_       = Tensor(1, max_steps_);

    // h_0 = 0
    for (int h = 0; h < hidden_size_; ++h) hidden_states_[0][h] = 0.0;

    double N = 0.0;      // running halting probability sum
    halt_step_ = max_steps_ - 1;  // default: never halt, stop at last step
    bool halted = false;
    total_ponder_ = 0.0;

    for (int t = 0; t < max_steps_; ++t) {
        // Pull x_t (1, input_dim) from cached inputs.
        // x_t[i] = inputs_[0][t * input_dim + i]  (row-vector layout)

        // Compute pre-h = x_t · W_xh^T + h_{t-1} · W_hh^T + b
        //   W_xh is (hidden, input), so x_t * W_xh^T = (1, hidden)
        //   h_{t-1} is (1, hidden), W_hh is (hidden, hidden), so h_{t-1} * W_hh^T = (1, hidden)
        // We avoid tensor multiplications to keep gradient-checking clean.
        for (int hp = 0; hp < hidden_size_; ++hp) {
            double pre = b_[hp][0];
            for (int d = 0; d < input_dim_; ++d) {
                pre += input[0][t * input_dim_ + d] * W_xh_[hp][d];
            }
            // h_{t-1} row at index t-1 in hidden_states_
            const double* h_prev = hidden_states_[t];
            for (int k = 0; k < hidden_size_; ++k) {
                pre += h_prev[k] * W_hh_[hp][k];
            }
            double h_t = std::tanh(pre);
            hidden_states_[t + 1][hp] = h_t;
            outputs_[t][hp] = h_t;     // y_t = h_t
        }

        // Compute halting logit z_t = y_t · W_halt + b_halt
        double z = b_halt_[0][0];
        const double* y_t = outputs_[t];
        for (int hp = 0; hp < hidden_size_; ++hp) {
            z += y_t[hp] * W_halt_[hp][0];
        }
        halt_logits_[0][t] = z;
        double p = 1.0 / (1.0 + std::exp(-z));
        halt_probs_[0][t] = p;

        if (halted) {
            // We already halted at an earlier step; this step produces no output.
            weights_[0][t] = 0.0;
            continue;
        }

        // Determine weight for this step.
        // Two cases for halting:
        // (a) Sum crosses 1: halt here, weight = 1 - N (remainder).
        // (b) Last step and we never crossed 1: halt here, weight = 1 - N (remainder).
        // Otherwise: this step's weight is p_t (regular step).
        bool crosses_one = (N + p >= 1.0 && N < 1.0);
        bool last_step = (t == max_steps_ - 1);
        if (crosses_one || last_step) {
            // Halt here.
            weights_[0][t] = 1.0 - N;
            halt_step_ = t;
            halted = true;
            total_ponder_ += p;
            // Subsequent steps: zero weight (loop body still runs to update
            // the y_t cache and halt_probs_/halt_logits_ caches for the
            // backward, but they contribute no gradient).
            for (int t2 = t + 1; t2 < max_steps_; ++t2) {
                weights_[0][t2] = 0.0;
            }
        } else {
            // Regular step (not the halt step).
            weights_[0][t] = p;
            total_ponder_ += p;
            N += p;
        }
    }

    // 3. Output = sum_t w_t * y_t
    Tensor output(1, hidden_size_);
    output.fill(0.0);
    for (int t = 0; t < max_steps_; ++t) {
        double w = weights_[0][t];
        if (w == 0.0) continue;
        const double* y = outputs_[t];
        for (int hp = 0; hp < hidden_size_; ++hp) {
            output[0][hp] += w * y[hp];
        }
    }
    return output;
}

Tensor ACTRecurrentLayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != 1 || grad_output.cols != static_cast<size_t>(hidden_size_)) {
        throw std::invalid_argument("ACTRecurrentLayer::backward: grad_output shape mismatch");
    }

    // Zero gradients.
    grad_W_xh_.fill(0.0);
    grad_W_hh_.fill(0.0);
    grad_b_.fill(0.0);
    grad_W_halt_.fill(0.0);
    grad_b_halt_.fill(0.0);

    // Per-step gradient accumulators.
    // d_y[t] = gradient of the loss w.r.t. y_t (shape: (1, hidden))
    Tensor d_y(max_steps_, hidden_size_);
    d_y.fill(0.0);

    // d_p[t] = gradient of the loss w.r.t. p_t (scalar per step)
    std::vector<double> d_p(max_steps_, 0.0);

    // d_z[t] = gradient w.r.t. the pre-sigmoid logit (scalar per step)
    std::vector<double> d_z(max_steps_, 0.0);

    // Step 1: Direct contribution from the weighted-output formula.
    //   For each t: d_y[t] += w_t * grad_output
    //              d_w[t]  = <y_t, grad_output>  (dot product)
    std::vector<double> d_w(max_steps_, 0.0);
    for (int t = 0; t < max_steps_; ++t) {
        double w = weights_[0][t];
        for (int hp = 0; hp < hidden_size_; ++hp) {
            d_y[t][hp] += w * grad_output[0][hp];
            d_w[t] += outputs_[t][hp] * grad_output[0][hp];
        }
    }

    // Step 2: For t < halt_step, w_t = p_t (direct derivative).
    for (int t = 0; t < halt_step_; ++t) {
        d_p[t] += d_w[t];
    }
    // For t = halt_step, w_t = 1 - N_{t-1}, where N_{t-1} = sum_{s <= t-1} p_s.
    //   So dL/d p_s = dL/d w_halt * (-1) = -d_w[halt] for each s <= halt-1.
    //   Each p_s appears in N_{t-1} exactly ONCE (not "halt - s" times — that
    //   confuses N_{t-1} with the running sum at intermediate halts; in the
    //   single-halt ACT setting N_{t-1} is just the plain sum).
    double chain_factor = -d_w[halt_step_];
    for (int s = 0; s < halt_step_; ++s) {
        d_p[s] += chain_factor;
    }

    // Step 3: Ponder penalty contribution.
    // The loss includes ponder_penalty * Σ_t p_t for t in [0, halt_step].
    // (The paper sums p_t over all steps up to and including the halt step;
    //  we follow the same convention. For t > halt_step, no contribution.)
    for (int t = 0; t <= halt_step_; ++t) {
        d_p[t] += ponder_penalty_;
    }

    // Step 4: Convert d_p[t] to d_z[t] via sigmoid derivative.
    for (int t = 0; t <= halt_step_; ++t) {
        double p = halt_probs_[0][t];
        d_z[t] = d_p[t] * p * (1.0 - p);
    }

    // Step 5: Backprop through the halting head.
    //   d_W_halt += y_t ⊗ d_z[t]   (outer product; shape (hidden, 1))
    //   d_b_halt += d_z[t]
    //   d_y[t]  += d_z[t] * W_halt  (broadcast scalar over hidden dim)
    for (int t = 0; t <= halt_step_; ++t) {
        const double* y = outputs_[t];
        for (int hp = 0; hp < hidden_size_; ++hp) {
            grad_W_halt_[hp][0] += y[hp] * d_z[t];
            d_y[t][hp] += d_z[t] * W_halt_[hp][0];
        }
        grad_b_halt_[0][0] += d_z[t];
    }

    // Step 6: BPTT through the tanh RNN from t = halt_step down to 0.
    //   The per-step d_y[t] we computed above is the upstream gradient into
    //   h_t = y_t (because y_t = h_t). The recurrence is:
    //     h_t = tanh(pre_h_t), pre_h_t = x_t · W_xh^T + h_{t-1} · W_hh^T + b
    //   Standard BPTT.
    Tensor grad_h_next(1, hidden_size_);  // gradient flowing into h_t from later steps (t+1, t+2, ...)
    grad_h_next.fill(0.0);
    Tensor grad_h(1, hidden_size_);       // gradient into pre_h_t this step (= dL/d pre_h_t)
    grad_h.fill(0.0);
    Tensor prev_grad_h(1, hidden_size_);  // grad_h from the PREVIOUS iteration (= dL/d pre_h_{t+1}); used to compute grad_h_next for the next iteration
    prev_grad_h.fill(0.0);

    Tensor grad_input(1, max_steps_ * input_dim_);
    grad_input.fill(0.0);

    for (int t = halt_step_; t >= 0; --t) {
        // grad_h_next at the top of this iteration = dL/d h_t from later steps
        // (set at the end of iteration t+1). At t = halt_step_, there is no later
        // step, so grad_h_next = 0.
        // Combine upstream from output side (d_y[t]) and from h_{t+1}.
        for (int hp = 0; hp < hidden_size_; ++hp) {
            grad_h[0][hp] = d_y[t][hp] + grad_h_next[0][hp];
        }
        // d/d pre_h = grad_h * (1 - h_t^2); h_t = hidden_states_[t+1].
        for (int hp = 0; hp < hidden_size_; ++hp) {
            double h_t = hidden_states_[t + 1][hp];
            double dtanh = 1.0 - h_t * h_t;
            grad_h[0][hp] *= dtanh;
        }
        // grad_h is now dL/d pre_h_t for this step. Accumulate parameter gradients.
        for (int hp = 0; hp < hidden_size_; ++hp) {
            double gh = grad_h[0][hp];
            for (int d = 0; d < input_dim_; ++d) {
                grad_W_xh_[hp][d] += gh * inputs_[0][t * input_dim_ + d];
                grad_input[0][t * input_dim_ + d] += gh * W_xh_[hp][d];
            }
            for (int k = 0; k < hidden_size_; ++k) {
                grad_W_hh_[hp][k] += gh * hidden_states_[t][k];
            }
            grad_b_[hp][0] += gh;
        }
        // Prepare grad_h_next for the next iteration (t-1):
        //   dL/d h_t[k] = sum_hp (dL/d pre_h_t[hp]) * W_hh[hp][k] * (1 - h_t^2[hp]) / (1 - h_t^2[hp])
        //              = sum_hp grad_h[hp] * W_hh[hp][k]
        // OVERWRITE grad_h_next (do not accumulate) — grad_h_next at the top of THIS
        // iteration held dL/d h_t from later steps, which we have already consumed.
        // For iteration t-1 we want a fresh dL/d h_{t-1} from t onward.
        for (int k = 0; k < hidden_size_; ++k) {
            double s = 0.0;
            for (int hp = 0; hp < hidden_size_; ++hp) {
                s += grad_h[0][hp] * W_hh_[hp][k];
            }
            grad_h_next[0][k] = s;
        }
        (void)prev_grad_h;  // reserved for future use; not needed in this form
    }

    return grad_input;
}

void ACTRecurrentLayer::update_weights(double learning_rate) {
    // SGD update on all 5 parameter groups.
    for (int hp = 0; hp < hidden_size_; ++hp) {
        for (int d = 0; d < input_dim_; ++d) {
            W_xh_[hp][d] -= learning_rate * grad_W_xh_[hp][d];
        }
        for (int k = 0; k < hidden_size_; ++k) {
            W_hh_[hp][k] -= learning_rate * grad_W_hh_[hp][k];
        }
        b_[hp][0] -= learning_rate * grad_b_[hp][0];
        W_halt_[hp][0] -= learning_rate * grad_W_halt_[hp][0];
    }
    b_halt_[0][0] -= learning_rate * grad_b_halt_[0][0];
}

void ACTRecurrentLayer::zero_grad() {
    grad_W_xh_.fill(0.0);
    grad_W_hh_.fill(0.0);
    grad_b_.fill(0.0);
    grad_W_halt_.fill(0.0);
    grad_b_halt_.fill(0.0);
}

Tensor ACTRecurrentLayer::get_weights() const {
    // Flatten all learnable parameters into one tensor. We just return W_xh for simplicity.
    return W_xh_;
}

Tensor ACTRecurrentLayer::get_gradients() const {
    return grad_W_xh_;
}

std::vector<Tensor*> ACTRecurrentLayer::parameters() {
    return {&W_xh_, &W_hh_, &b_, &W_halt_, &b_halt_};
}

std::vector<Tensor*> ACTRecurrentLayer::gradients() {
    return {&grad_W_xh_, &grad_W_hh_, &grad_b_, &grad_W_halt_, &grad_b_halt_};
}
