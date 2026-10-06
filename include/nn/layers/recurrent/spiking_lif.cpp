#include "nn/layers/recurrent/spiking_lif.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

// ============================================================================
// Spiking LIF neurons with surrogate gradients.
// Math and conventions documented in the header; Eq. 4-5 and Sec. V-B of
// arXiv:1901.09948v2.
// ============================================================================

// ----------------------------------------------------------------------------
// LIFNeuron — constructor
// ----------------------------------------------------------------------------
LIFNeuron::LIFNeuron(size_t input_dim, size_t hidden_size,
                     double alpha, double beta,
                     double threshold, double gamma,
                     bool use_recurrence)
    : input_dim_(input_dim), hidden_size_(hidden_size),
      alpha_(alpha), beta_(beta), threshold_(threshold),
      use_recurrence_(use_recurrence) {

    if (input_dim == 0)
        throw std::invalid_argument("LIFNeuron requires input_dim > 0");
    if (hidden_size == 0)
        throw std::invalid_argument("LIFNeuron requires hidden_size > 0");
    if (!(alpha > 0.0 && alpha < 1.0))
        throw std::invalid_argument("LIFNeuron requires 0 < alpha < 1");
    if (!(beta > 0.0 && beta < 1.0))
        throw std::invalid_argument("LIFNeuron requires 0 < beta < 1");
    if (!(threshold > 0.0))
        throw std::invalid_argument("LIFNeuron requires threshold > 0");
    if (!(gamma > 0.0))
        throw std::invalid_argument("LIFNeuron requires gamma > 0");

    W_    = Tensor::random(hidden_size, input_dim, 0.3);
    b_    = Tensor::zeros(hidden_size, 1);
    gamma_ = Tensor(1, 1);
    gamma_[0][0] = gamma;

    grad_W_     = Tensor::zeros(hidden_size, input_dim);
    grad_b_     = Tensor::zeros(hidden_size, 1);
    grad_gamma_ = Tensor::zeros(1, 1);

    if (use_recurrence_) {
        V_ = Tensor::random(hidden_size, hidden_size, 0.2);
        grad_V_ = Tensor::zeros(hidden_size, hidden_size);
    } else {
        V_ = Tensor::zeros(0, 0);
        grad_V_ = Tensor::zeros(0, 0);
    }
}

// ----------------------------------------------------------------------------
// LIFNeuron — forward
//
//   I_t[i] = α·I_{t-1}[i] + Σ_k W[i,k]·X[t,k] + b[i]  (+ Σ_j V[i,j]·S_{t-1}[j])
//   U_t[i] = β·U_{t-1}[i] + I_t[i] − ϑ·S_{t-1}[i]
//   S_t[i] = Θ(U_t[i] − ϑ)          with Θ(0) = 1
//
// NOTE the reset uses S_{t-1} (the previous step's spike), matching Eq. 5.
// ----------------------------------------------------------------------------
Tensor LIFNeuron::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("LIFNeuron forward: input cols mismatch");

    const size_t T = input.rows;
    const size_t H = hidden_size_;
    const size_t D = input_dim_;

    cache_X_ = input.clone();
    cache_I_ = Tensor::zeros(T, H);
    cache_U_ = Tensor::zeros(T, H);
    cache_S_ = Tensor::zeros(T, H);

    // Rolling state, shaped (H, 1) to match the [i][0] indexing used below.
    Tensor I_prev = Tensor::zeros(H, 1);
    Tensor U_prev = Tensor::zeros(H, 1);
    Tensor S_prev = Tensor::zeros(H, 1);

    for (size_t t = 0; t < T; ++t) {
        // --- Eq. 4: synaptic current ---
        for (size_t i = 0; i < H; ++i) {
            double acc = alpha_ * I_prev[i][0];
            for (size_t k = 0; k < D; ++k)
                acc += W_[i][k] * input[t][k];
            acc += b_[i][0];
            if (use_recurrence_) {
                for (size_t j = 0; j < H; ++j)
                    acc += V_[i][j] * S_prev[j][0];
            }
            cache_I_[t][i] = acc;
        }

        // --- Eq. 5: membrane, with reset driven by the PREVIOUS spike ---
        for (size_t i = 0; i < H; ++i) {
            double acc = beta_ * U_prev[i][0] + cache_I_[t][i]
                       - threshold_ * S_prev[i][0];
            cache_U_[t][i] = acc;
            cache_S_[t][i] = SurrogateSpike::spike(acc, threshold_);
        }

        // Roll state
        for (size_t i = 0; i < H; ++i) {
            I_prev[i][0] = cache_I_[t][i];
            U_prev[i][0] = cache_U_[t][i];
            S_prev[i][0] = cache_S_[t][i];
        }
    }

    return cache_S_.clone();
}

// ----------------------------------------------------------------------------
// LIFNeuron — backward (BPTT with surrogate substitution)
//
//   dU_t[i]  = g_t[i] · 1/(γ|U_t[i]−ϑ|+1)      <- surrogate for ∂S/∂U
//   dI_t[i]  = dU_t[i]
//   dI_{t-1} = α · dI_t[i]
//   dU_{t-1} = β · dU_t[i] + α · dI_t[i]        <- NO reset term: −ϑ·S_{t-1} is
//                                               constant in U and in all params
//   grad_W[i,k] += dI_t[i]·X[t,k]
//   grad_b[i]   += dI_t[i]
//   grad_V[i,j] += dI_t[i]·S_{t-1}[j]            (recurrent)
//   dX[t,k]    += Σ_i dI_t[i]·W[i,k]
//   grad_gamma  += Σ_i g_t[i]·(−|U_t[i]−ϑ|)/(γ|U_t[i]−ϑ|+1)²
// ----------------------------------------------------------------------------
Tensor LIFNeuron::backward(const Tensor& grad_output, double /* learning_rate */) {
    if (grad_output.rows != cache_S_.rows || grad_output.cols != cache_S_.cols)
        throw std::invalid_argument("LIFNeuron backward: grad_output shape mismatch");

    const size_t T = cache_S_.rows;
    const size_t H = hidden_size_;
    const size_t D = input_dim_;

    Tensor grad_X = Tensor::zeros(T, D);
    const double gam = gamma_[0][0];

    // BPTT over the spike train, accumulating two adjoints backwards:
    //
    //   dI_prev[i] : dL/dI_{t}   carried in from step t+1 via Eq. 4's alpha
    //   dS_prev[i] : dL/dS_{t}   carried in from step t+1, which reaches
    //                             U_{t+1} through Eq. 5's -theta*S_t reset
    //
    // Per step, per neuron i:
    //   dU_t[i] =  g_t[i]·F(U_t[i])                       (direct, surrogate)
    //            - theta·dS_prev[i]·F(U_{t-1}[i])         (reset feedback)
    //   dI_t[i] =  dU_t[i] + alpha·dI_prev[i]
    //
    // The reset feedback is a real gradient path even though -theta*S is a
    // constant w.r.t. U, because S_{t-1} = F(U_{t-1}) itself has the surrogate
    // derivative. It is ALSO the term FD cannot check: the true dS/dU is zero
    // everywhere, so finite differences of the hard loss are identically 0 for
    // a correct and an incorrect backward alike. Verified instead against the
    // adjoint of the smoothed model in tests/test_spiking_lif.cpp.
    //
    // NOTE the index: the reset of neuron i is affected ONLY by neuron i's own
    // previous spike, so dS_prev is indexed [i], not summed over j.
    Tensor dI_prev = Tensor::zeros(H, 1);
    Tensor dS_prev = Tensor::zeros(H, 1);

    for (size_t ti = T; ti-- > 0; ) {
        const size_t t = ti;

        Tensor dU = Tensor::zeros(H, 1);
        Tensor dI = Tensor::zeros(H, 1);

        for (size_t i = 0; i < H; ++i) {
            const double g = grad_output[t][i];

            // (1) Direct path: S_t = theta(U_t - theta), surrogate-substituted.
            dU[i][0] = g * SurrogateSpike::surrogate_derivative(cache_U_[t][i],
                                                               threshold_, gam);
            grad_gamma_[0][0] += g * SurrogateSpike::surrogate_derivative_dgamma(
                                            cache_U_[t][i], threshold_, gam);

            // (2) Reset feedback: S_{t-1} = F(U_{t-1}) enters U_t with -theta.
            if (t > 0) {
                const double f_prev = SurrogateSpike::surrogate_derivative(
                                           cache_U_[t-1][i], threshold_, gam);
                dU[i][0] += (-threshold_) * dS_prev[i][0] * f_prev;
                grad_gamma_[0][0] += (-threshold_) * dS_prev[i][0]
                    * SurrogateSpike::surrogate_derivative_dgamma(cache_U_[t-1][i],
                                                                  threshold_, gam);
            }

            // (3) I_t -> U_t (coefficient 1) and I_t -> I_{t-1} (coefficient alpha).
            dI[i][0] = dU[i][0] + alpha_ * dI_prev[i][0];

            // (4) Parameter gradients and the input gradient.
            grad_b_[i][0] += dI[i][0];
            for (size_t k = 0; k < D; ++k) {
                grad_W_[i][k] += dI[i][0] * cache_X_[t][k];
                grad_X[t][k]   += dI[i][0] * W_[i][k];
            }
            if (use_recurrence_ && t > 0) {
                for (size_t j = 0; j < H; ++j)
                    grad_V_[i][j] += dI[i][0] * cache_S_[t - 1][j];
            }
        }

        // Carry to step t-1.
        for (size_t i = 0; i < H; ++i) {
            dI_prev[i][0] = dI[i][0];
            dS_prev[i][0] = grad_output[t][i];
        }
    }

    return grad_X;
}

// ----------------------------------------------------------------------------
// LIFNeuron — parameter update / grad plumbing
// ----------------------------------------------------------------------------
void LIFNeuron::update_weights(double lr) {
    for (size_t i = 0; i < W_.rows; ++i)
        for (size_t k = 0; k < W_.cols; ++k)
            W_[i][k] -= lr * grad_W_[i][k];
    for (size_t i = 0; i < b_.cols; ++i)
        b_[i][0] -= lr * grad_b_[i][0];
    gamma_[0][0] -= lr * grad_gamma_[0][0];
    if (use_recurrence_) {
        for (size_t i = 0; i < V_.rows; ++i)
            for (size_t j = 0; j < V_.cols; ++j)
                V_[i][j] -= lr * grad_V_[i][j];
    }
}

void LIFNeuron::zero_grad() {
    grad_W_ = Tensor::zeros(hidden_size_, input_dim_);
    grad_b_ = Tensor::zeros(hidden_size_, 1);
    grad_gamma_ = Tensor::zeros(1, 1);
    if (use_recurrence_)
        grad_V_ = Tensor::zeros(hidden_size_, hidden_size_);
}

std::vector<Tensor*> LIFNeuron::parameters() {
    if (use_recurrence_)
        return {&W_, &b_, &gamma_, &V_};
    return {&W_, &b_, &gamma_};
}

std::vector<Tensor*> LIFNeuron::gradients() {
    if (use_recurrence_)
        return {&grad_W_, &grad_b_, &grad_gamma_, &grad_V_};
    return {&grad_W_, &grad_b_, &grad_gamma_};
}

// ----------------------------------------------------------------------------
// SpikingNet
// ----------------------------------------------------------------------------
SpikingNet::SpikingNet(size_t input_dim, size_t hidden_size, size_t output_dim,
                       size_t num_layers,
                       double alpha, double beta,
                       double threshold, double gamma,
                       double encode_frac, double encode_gain)
    : input_dim_(input_dim), hidden_size_(hidden_size), output_dim_(output_dim),
      alpha_(alpha), beta_(beta), threshold_(threshold), gamma_(gamma),
      encode_frac_(encode_frac), encode_gain_(encode_gain) {

    if (input_dim == 0)
        throw std::invalid_argument("SpikingNet requires input_dim > 0");
    if (hidden_size == 0)
        throw std::invalid_argument("SpikingNet requires hidden_size > 0");
    if (output_dim == 0)
        throw std::invalid_argument("SpikingNet requires output_dim > 0");
    if (num_layers == 0)
        throw std::invalid_argument("SpikingNet requires num_layers > 0");
    if (!(encode_frac > 0.0 && encode_frac < 1.0))
        throw std::invalid_argument("SpikingNet requires 0 < encode_frac < 1");
    if (!(encode_gain > 0.0))
        throw std::invalid_argument("SpikingNet requires encode_gain > 0");

    for (size_t l = 0; l < num_layers; ++l) {
        size_t in_dim = (l == 0) ? input_dim : hidden_size;
        layers_.push_back(std::unique_ptr<LIFNeuron>(
            new LIFNeuron(in_dim, hidden_size, alpha, beta, threshold, gamma)));
    }
    readout_ = std::unique_ptr<Dense>(new Dense(hidden_size, output_dim));
}

// Rate encoding: emit a spike on the SIGNED value exceeding a per-row
// magnitude threshold, so opposite-sign inputs produce DIFFERENT spike trains
// (ranking by |x| alone is sign-blind: [0.5,-0.3] and [-0.5,0.3] would encode
// identically and no downstream layer could separate them).
//
// The threshold is strictly below each row's max magnitude (frac < 1) so the
// strongest feature always emits a spike — an encoder whose threshold sat AT
// the max would emit nothing and the whole network would be dead.
//
// GAIN matters as much as the threshold: the spike train is 0/1, so the input
// current into layer 0 is at most sum_j |W[0,j]|, and it must be able to reach
// the threshold. That is why the readout scale `encode_gain_` multiplies the
// binary train before it enters the LIF stack; Test 9b pins that a spike
// actually reaches the threshold.
Tensor SpikingNet::encode_rate(const Tensor& input) {
    const size_t B = input.rows;
    const size_t D = input.cols;
    Tensor E = Tensor::zeros(B, D);
    for (size_t b = 0; b < B; ++b) {
        double mx = 0.0;
        for (size_t j = 0; j < D; ++j) mx = std::max(mx, std::abs(input[b][j]));
        if (mx <= 0.0) continue;
        const double thr = encode_frac_ * mx;
        for (size_t j = 0; j < D; ++j)
            E[b][j] = (input[b][j] > thr) ? 1.0 : 0.0;
    }
    return E;
}

Tensor SpikingNet::forward(const Tensor& input) {
    if (input.cols != input_dim_)
        throw std::invalid_argument("SpikingNet forward: input cols mismatch");

    cache_encoded_ = encode_rate(input);
    cache_layer_out_.clear();

    Tensor cur = cache_encoded_.clone();
    for (size_t l = 0; l < layers_.size(); ++l) {
        if (l == 0 && encode_gain_ != 1.0) {
            // Scale the binary train so the induced input current can reach
            // the firing threshold; without this the LIF layer can never fire.
            for (size_t i = 0; i < cur.rows * cur.cols; ++i)
                cur.data[i] *= encode_gain_;
        }
        cur = layers_[l]->forward(cur);
        cache_layer_out_.push_back(cur.clone());
    }

    // Readout consumes the final layer's spike train directly.
    return readout_->forward(cur);
}

Tensor SpikingNet::backward(const Tensor& grad_output, double lr) {
    Tensor g = readout_->backward(grad_output, lr);

    // Walk the LIF stack backwards. LIFNeuron::backward returns the gradient
    // w.r.t. its own input, which is exactly the next (earlier) layer's
    // grad_output.
    for (size_t li = layers_.size(); li-- > 0; )
        g = layers_[li]->backward(g, lr);

    // The rate encoder is a hard threshold (piecewise-constant in the input),
    // so no gradient reaches the caller's tensor. Returning zeros rather than
    // a silent partial is the honest answer; a ANN-to-SNN conversion front-end
    // would supply the input gradient itself.
    return Tensor::zeros(cache_encoded_.rows, input_dim_);
}

void SpikingNet::update_weights(double lr) {
    for (auto& l : layers_) l->update_weights(lr);
    readout_->update_weights(lr);
}

void SpikingNet::zero_grad() {
    for (auto& l : layers_) l->zero_grad();
    readout_->zero_grad();
}

std::vector<Tensor*> SpikingNet::parameters() {
    std::vector<Tensor*> out;
    for (auto& l : layers_) {
        std::vector<Tensor*> p = l->parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    out.push_back(&readout_->weights);
    out.push_back(&readout_->bias);
    return out;
}

std::vector<Tensor*> SpikingNet::gradients() {
    std::vector<Tensor*> out;
    for (auto& l : layers_) {
        std::vector<Tensor*> g = l->gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    out.push_back(&readout_->grad_weights);
    out.push_back(&readout_->grad_bias);
    return out;
}

Tensor SpikingNet::get_weights() const { return readout_->weights; }
Tensor SpikingNet::get_gradients() const { return readout_->grad_weights; }

Tensor SpikingNet::layer_spikes(size_t l) const {
    if (l >= cache_layer_out_.size()) return Tensor();
    return cache_layer_out_[l];
}