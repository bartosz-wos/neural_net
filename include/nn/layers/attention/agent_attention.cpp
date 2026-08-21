// ============================================================================
// Agent Attention — Han et al. 2024 (ECCV 2024)
//   "Agent Attention: On the Integration of Softmax and Linear Attention"
//   https://arxiv.org/abs/2312.08874
//
// See agent_attention.h for the full mathematical formulation. This file
// implements:
//   * AgentAttention       — single-head agent attention with optional
//                            linear residual (DABA-style)
//   * AgentAttentionBlock  — pre-LN → AgentAttention → residual →
//                            pre-LN → FFN (GELU) → residual
//   * AgentAttentionModel  — stack of AgentAttentionBlocks + classifier
//
// Conventions match the rest of the repo (Performer / Linformer / AFT):
//   * Dense: y = X @ W^T + b, W stored as (out, in).
//   * (N, d_model) input/output, row-major.
//   * Single-head; for multi-head, call multiple AgentAttentions and concat.
// ============================================================================

#include "agent_attention.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>

// Softplus — used to map log_lambda (unconstrained) to λ ≥ 0.
static inline double softplus(double x) {
    // Numerically stable: log(1 + exp(x))
    if (x > 30.0) return x;
    if (x < -30.0) return 0.0;
    return std::log1p(std::exp(x));
}
static inline double softplus_deriv(double x) {
    // d softplus(x) / dx = sigmoid(x)
    if (x > 0) return 1.0 / (1.0 + std::exp(-x));
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

// Row-wise softmax: writes softmax into the same matrix (in-place).
// Each row is normalized independently. Scale is the pre-factor (typically 1/sqrt(d)).
static void row_softmax_inplace(Tensor& A, double scale) {
    const size_t n = A.rows;
    const size_t m = A.cols;
    for (size_t i = 0; i < n; ++i) {
        // 1. Apply scale and find max for stability
        double max_val = -1e300;
        for (size_t j = 0; j < m; ++j) {
            A[i][j] *= scale;
            if (A[i][j] > max_val) max_val = A[i][j];
        }
        // 2. exp and sum
        double sum = 0.0;
        for (size_t j = 0; j < m; ++j) {
            A[i][j] = std::exp(A[i][j] - max_val);
            sum += A[i][j];
        }
        // 3. Normalize
        const double inv = 1.0 / sum;
        for (size_t j = 0; j < m; ++j) A[i][j] *= inv;
    }
}

// ============================================================================
// AgentAttention
// ============================================================================

AgentAttention::AgentAttention(size_t d_model, size_t num_agents,
                               bool use_linear_residual)
    : W_q(d_model, d_model),
      W_k(d_model, d_model),
      W_v(d_model, d_model),
      W_o(d_model, d_model),
      W_q_agents(d_model, d_model),
      W_k_agents(d_model, d_model),
      agents_(num_agents, d_model),
      grad_agents_(num_agents, d_model),
      log_lambda_(1, 1),
      grad_log_lambda_(1, 1),
      d_model_(d_model),
      num_agents_(num_agents),
      use_linear_residual_(use_linear_residual)
{
    if (d_model == 0) {
        throw std::invalid_argument("AgentAttention: d_model must be > 0");
    }
    if (num_agents == 0) {
        throw std::invalid_argument("AgentAttention: num_agents must be > 0");
    }

    // Initialize agents to small random values so they don't start at zero.
    std::mt19937 gen(123);
    std::normal_distribution<> dis(0.0, 0.05);
    for (size_t i = 0; i < num_agents_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            agents_(i, j) = dis(gen);

    // log_lambda = 0  → λ = softplus(0) = log(2) ≈ 0.693
    log_lambda_(0, 0) = 0.0;
    grad_log_lambda_.fill(0.0);
    grad_agents_.fill(0.0);
}

Tensor AgentAttention::forward(const Tensor& input) {
    const size_t N = input.rows;
    if (input.cols != d_model_) {
        throw std::invalid_argument("AgentAttention: input.cols must equal d_model");
    }
    N_last_ = N;

    last_input_ = input.clone();

    // Project to Q, K, V via Dense (y = X @ W^T + b).
    last_q_ = W_q.forward(input);    // (N, d_model)
    last_k_ = W_k.forward(input);    // (N, d_model)
    last_v_ = W_v.forward(input);    // (N, d_model)

    // Project agents to Q_A and K_A via their own Dense projections.
    last_qa_ = W_q_agents.forward(agents_);  // (N_a, d_model)
    last_ka_ = W_k_agents.forward(agents_);  // (N_a, d_model)

    // Cache raw (un-scaled) attn scores for backward.
    // Stage 1: agent aggregation. S_agg = Q_A · K^T  (N_a × N)
    Tensor S_agg(num_agents_, N);
    for (size_t a = 0; a < num_agents_; ++a)
        for (size_t s = 0; s < N; ++s) {
            double s_val = 0.0;
            for (size_t j = 0; j < d_model_; ++j)
                s_val += last_qa_(a, j) * last_k_(s, j);
            S_agg(a, s) = s_val;
        }
    last_attn_agg_ = S_agg.clone();  // pre-softmax; will be overwritten with post-softmax
    const double scale = 1.0 / std::sqrt(static_cast<double>(d_model_));
    row_softmax_inplace(S_agg, scale);  // Now S_agg[i] is the row-softmax
    last_attn_agg_ = S_agg;  // overwrite with the post-softmax — used as P_agg going forward

    // A' = P_agg · V  (N_a × d_model)
    last_A_prime_ = Tensor(num_agents_, d_model_);
    for (size_t a = 0; a < num_agents_; ++a)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t s = 0; s < N; ++s)
                acc += last_attn_agg_(a, s) * last_v_(s, j);
            last_A_prime_(a, j) = acc;
        }

    // Stage 2: agent broadcast. S_brd = Q · K_A^T (N × N_a)
    Tensor S_brd(N, num_agents_);
    for (size_t t = 0; t < N; ++t)
        for (size_t a = 0; a < num_agents_; ++a) {
            double s_val = 0.0;
            for (size_t j = 0; j < d_model_; ++j)
                s_val += last_q_(t, j) * last_ka_(a, j);
            S_brd(t, a) = s_val;
        }
    last_attn_brd_ = S_brd.clone();  // post-softmax
    row_softmax_inplace(S_brd, scale);  // S_brd[:, a] is the row-softmax → overwrite
    last_attn_brd_ = S_brd;  // Now last_attn_brd_ is the post-softmax (P_brd)

    // O = P_brd · A'  (N × d_model)
    last_O_agent_ = Tensor(N, d_model_);
    for (size_t t = 0; t < N; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t a = 0; a < num_agents_; ++a)
                acc += last_attn_brd_(t, a) * last_A_prime_(a, j);
            last_O_agent_(t, j) = acc;
        }

    // Linear residual (DABA): O_lin = softmax(Q · K^T / sqrt(d)) · V
    if (use_linear_residual_) {
        Tensor S_lin(N, N);
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s) {
                double s_val = 0.0;
                for (size_t j = 0; j < d_model_; ++j)
                    s_val += last_q_(t, j) * last_k_(s, j);
                S_lin(t, s) = s_val;
            }
        last_attn_lin_ = S_lin.clone();
        row_softmax_inplace(S_lin, scale);
        last_attn_lin_ = S_lin;

        last_O_lin_ = Tensor(N, d_model_);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d_model_; ++j) {
                double acc = 0.0;
                for (size_t s = 0; s < N; ++s)
                    acc += last_attn_lin_(t, s) * last_v_(s, j);
                last_O_lin_(t, j) = acc;
            }
    } else {
        last_O_lin_ = Tensor(N, d_model_);
        last_O_lin_.fill(0.0);
    }

    // Combine: O_final = O_agent + λ · O_lin
    last_lambda_ = softplus(log_lambda_(0, 0));
    last_O_final_ = Tensor(N, d_model_);
    for (size_t t = 0; t < N; ++t)
        for (size_t j = 0; j < d_model_; ++j)
            last_O_final_(t, j) = last_O_agent_(t, j) + last_lambda_ * last_O_lin_(t, j);

    // Output projection: y = O_final @ W_o^T + b_o
    Tensor output = W_o.forward(last_O_final_);
    return output;
}

Tensor AgentAttention::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (grad_output.rows != N_last_ || grad_output.cols != d_model_) {
        throw std::invalid_argument("AgentAttention::backward: grad_output shape mismatch");
    }

    const size_t N = N_last_;
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
    W_q_agents.zero_grad();
    W_k_agents.zero_grad();
    grad_agents_.fill(0.0);
    grad_log_lambda_.fill(0.0);

    const double scale = 1.0 / std::sqrt(static_cast<double>(d_model_));

    // Step 1: backward through W_o. y = O_final @ W_o^T + b_o.
    // Grad of O_final: grad_O_final[t, j] = sum_j' grad_out[t, j'] * W_o[j'][j]
    Tensor grad_O_final(N, d_model_);
    for (size_t t = 0; t < N; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t jp = 0; jp < d_model_; ++jp)
                acc += grad_output(t, jp) * W_o.weights(jp, j);
            grad_O_final(t, j) = acc;
        }
    // W_o gradients
    for (size_t j = 0; j < d_model_; ++j)
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t)
                acc += grad_output(t, j) * last_O_final_(t, i);
            W_o.grad_weights(j, i) += acc;
        }
    for (size_t j = 0; j < d_model_; ++j) {
        double b_acc = 0.0;
        for (size_t t = 0; t < N; ++t) b_acc += grad_output(t, j);
        W_o.grad_bias(0, j) += b_acc;
    }

    // Step 2: split grad_O_final into grad_O_agent and grad_O_lin (via λ)
    // O_final = O_agent + λ · O_lin
    Tensor grad_O_agent(N, d_model_);
    Tensor grad_O_lin(N, d_model_);
    for (size_t t = 0; t < N; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            grad_O_agent(t, j) = grad_O_final(t, j);
            grad_O_lin(t, j) = grad_O_final(t, j) * last_lambda_;
        }
    if (use_linear_residual_) {
        // grad_log_lambda = sum_{t,j} grad_O_final * (dλ/dlogλ) * O_lin
        // dλ/dlogλ = σ(log_lambda) = sigmoid(log_lambda)
        double dlam_dloglam = softplus_deriv(log_lambda_(0, 0));
        double grad_ll = 0.0;
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d_model_; ++j)
                grad_ll += grad_O_final(t, j) * dlam_dloglam * last_O_lin_(t, j);
        grad_log_lambda_(0, 0) = grad_ll;
    }

    // Step 3: backward through the linear residual (when active).
    // O_lin = P_lin · V, P_lin = softmax(Q K^T / sqrt(d))
    // (a) grad_P_lin = grad_O_lin · V^T  (N × N)
    // (b) softmax backward: grad_S_lin = P_lin ⊙ (grad_P_lin - rowsum(grad_P_lin ⊙ P_lin))
    // (c) grad_Q = grad_S_lin · K / sqrt(d)  (N × d); grad_K = grad_S_lin^T · Q / sqrt(d)  (N × d)
    // (d) grad_V = P_lin^T · grad_O_lin  (N × d)
    Tensor grad_Q(N, d_model_); grad_Q.fill(0.0);
    Tensor grad_K(N, d_model_); grad_K.fill(0.0);
    Tensor grad_V(N, d_model_); grad_V.fill(0.0);

    if (use_linear_residual_) {
        // (a) grad_P_lin
        Tensor grad_P_lin(N, N);
        for (size_t t = 0; t < N; ++t)
            for (size_t s = 0; s < N; ++s) {
                double acc = 0.0;
                for (size_t j = 0; j < d_model_; ++j)
                    acc += grad_O_lin(t, j) * last_v_(s, j);
                grad_P_lin(t, s) = acc;
            }
        // (b) softmax backward
        Tensor grad_S_lin(N, N);
        for (size_t t = 0; t < N; ++t) {
            double row_sum = 0.0;
            for (size_t s = 0; s < N; ++s)
                row_sum += grad_P_lin(t, s) * last_attn_lin_(t, s);
            for (size_t s = 0; s < N; ++s)
                grad_S_lin(t, s) = last_attn_lin_(t, s) * (grad_P_lin(t, s) - row_sum);
        }
        // (c) grad_Q += grad_S_lin · K · scale, grad_K += grad_S_lin^T · Q · scale
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d_model_; ++j) {
                double acc_q = 0.0, acc_k = 0.0;
                for (size_t s = 0; s < N; ++s) {
                    acc_q += grad_S_lin(t, s) * last_k_(s, j);
                    acc_k += grad_S_lin(s, t) * last_q_(s, j);
                }
                grad_Q(t, j) += scale * acc_q;
                grad_K(t, j) += scale * acc_k;
            }
        // (d) grad_V = P_lin^T · grad_O_lin
        for (size_t s = 0; s < N; ++s)
            for (size_t j = 0; j < d_model_; ++j) {
                double acc = 0.0;
                for (size_t t = 0; t < N; ++t)
                    acc += last_attn_lin_(t, s) * grad_O_lin(t, j);
                grad_V(s, j) += acc;
            }
    }

    // Step 4: backward through the agent stages.
    // Stage 2 (broadcast): O = P_brd · A'  (N × d)
    // (a) grad_P_brd = grad_O_agent · A'^T  (N × N_a)
    // (b) grad_A' += P_brd^T · grad_O_agent  (N_a × d)
    // (c) softmax backward: grad_S_brd = P_brd ⊙ (grad_P_brd - rowsum ⊙ P_brd)
    // (d) grad_Q += grad_S_brd · K_A · scale  (N × d)
    // (e) grad_K_A = Q^T · grad_S_brd · scale  (N_a × d)
    Tensor grad_P_brd(N, num_agents_);
    for (size_t t = 0; t < N; ++t)
        for (size_t a = 0; a < num_agents_; ++a) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j)
                acc += grad_O_agent(t, j) * last_A_prime_(a, j);
            grad_P_brd(t, a) = acc;
        }
    // grad_A' = P_brd^T · grad_O_agent
    Tensor grad_A_prime(num_agents_, d_model_);
    for (size_t a = 0; a < num_agents_; ++a)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t)
                acc += last_attn_brd_(t, a) * grad_O_agent(t, j);
            grad_A_prime(a, j) = acc;
        }
    // softmax backward over P_brd
    Tensor grad_S_brd(N, num_agents_);
    for (size_t t = 0; t < N; ++t) {
        double row_sum = 0.0;
        for (size_t a = 0; a < num_agents_; ++a)
            row_sum += grad_P_brd(t, a) * last_attn_brd_(t, a);
        for (size_t a = 0; a < num_agents_; ++a)
            grad_S_brd(t, a) = last_attn_brd_(t, a) * (grad_P_brd(t, a) - row_sum);
    }
    // grad_Q += grad_S_brd · K_A · scale
    for (size_t t = 0; t < N; ++t)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t a = 0; a < num_agents_; ++a)
                acc += grad_S_brd(t, a) * last_ka_(a, j);
            grad_Q(t, j) += scale * acc;
        }
    // grad_K_A = Q^T · grad_S_brd · scale
    Tensor grad_K_A(num_agents_, d_model_);
    for (size_t a = 0; a < num_agents_; ++a)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t)
                acc += last_q_(t, j) * grad_S_brd(t, a);
            grad_K_A(a, j) += scale * acc;
        }

    // Stage 1 (aggregation): A' = P_agg · V  (N_a × d)
    // (a) grad_P_agg = grad_A_prime · V^T  (N_a × N)
    // (b) grad_V += P_agg^T · grad_A_prime  (N × d)
    // (c) softmax backward: grad_S_agg = P_agg ⊙ (grad_P_agg - rowsum ⊙ P_agg)
    // (d) grad_Q_A = grad_S_agg · K · scale  (N_a × d)
    // (e) grad_K += grad_S_agg^T · Q_A · scale  (N × d)
    Tensor grad_P_agg(num_agents_, N);
    for (size_t a = 0; a < num_agents_; ++a)
        for (size_t s = 0; s < N; ++s) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j)
                acc += grad_A_prime(a, j) * last_v_(s, j);
            grad_P_agg(a, s) = acc;
        }
    // grad_V += P_agg^T · grad_A_prime
    for (size_t s = 0; s < N; ++s)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t a = 0; a < num_agents_; ++a)
                acc += last_attn_agg_(a, s) * grad_A_prime(a, j);
            grad_V(s, j) += acc;
        }
    // softmax backward
    Tensor grad_S_agg(num_agents_, N);
    for (size_t a = 0; a < num_agents_; ++a) {
        double row_sum = 0.0;
        for (size_t s = 0; s < N; ++s)
            row_sum += grad_P_agg(a, s) * last_attn_agg_(a, s);
        for (size_t s = 0; s < N; ++s)
            grad_S_agg(a, s) = last_attn_agg_(a, s) * (grad_P_agg(a, s) - row_sum);
    }
    // grad_Q_A = grad_S_agg · K · scale
    Tensor grad_Q_A(num_agents_, d_model_);
    for (size_t a = 0; a < num_agents_; ++a)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t s = 0; s < N; ++s)
                acc += grad_S_agg(a, s) * last_k_(s, j);
            grad_Q_A(a, j) += scale * acc;
        }
    // grad_K += grad_S_agg^T · Q_A · scale
    for (size_t s = 0; s < N; ++s)
        for (size_t j = 0; j < d_model_; ++j) {
            double acc = 0.0;
            for (size_t a = 0; a < num_agents_; ++a)
                acc += grad_S_agg(a, s) * last_qa_(a, j);
            grad_K(s, j) += scale * acc;
        }

    // Step 5: backward through W_q (grad_Q → W_q weights + bias + grad_input)
    // grad_Q is the gradient into Q. Q = X @ W_q^T + b_q. So grad_X (from W_q alone) =
    // grad_Q · W_q. Then we accumulate from W_k and W_v similarly.
    // First: deriv param gradients for W_q, W_k, W_v.
    // grad_W_q[j, i] += sum_t grad_Q[t, j] * X[t, i]
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t)
                acc += grad_Q(t, j) * last_input_(t, i);
            W_q.grad_weights(j, i) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < N; ++t) b_acc += grad_Q(t, j);
        W_q.grad_bias(0, j) += b_acc;
    }
    // grad_W_k[j, i] += sum_t grad_K[t, j] * X[t, i]
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t)
                acc += grad_K(t, j) * last_input_(t, i);
            W_k.grad_weights(j, i) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < N; ++t) b_acc += grad_K(t, j);
        W_k.grad_bias(0, j) += b_acc;
    }
    // grad_W_v[j, i] += sum_t grad_V[t, j] * X[t, i]
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t t = 0; t < N; ++t)
                acc += grad_V(t, j) * last_input_(t, i);
            W_v.grad_weights(j, i) += acc;
        }
        double b_acc = 0.0;
        for (size_t t = 0; t < N; ++t) b_acc += grad_V(t, j);
        W_v.grad_bias(0, j) += b_acc;
    }

    // Step 6: gradient w.r.t. W_q_agents, W_k_agents, agents_.
    // grad_W_q_agents[j, i] += sum_a grad_Q_A[a, j] * agents_[a, i]
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t a = 0; a < num_agents_; ++a)
                acc += grad_Q_A(a, j) * agents_(a, i);
            W_q_agents.grad_weights(j, i) += acc;
        }
        double b_acc = 0.0;
        for (size_t a = 0; a < num_agents_; ++a) b_acc += grad_Q_A(a, j);
        W_q_agents.grad_bias(0, j) += b_acc;
    }
    // grad_W_k_agents[j, i] += sum_a grad_K_A[a, j] * agents_[a, i]
    for (size_t j = 0; j < d_model_; ++j) {
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t a = 0; a < num_agents_; ++a)
                acc += grad_K_A(a, j) * agents_(a, i);
            W_k_agents.grad_weights(j, i) += acc;
        }
        double b_acc = 0.0;
        for (size_t a = 0; a < num_agents_; ++a) b_acc += grad_K_A(a, j);
        W_k_agents.grad_bias(0, j) += b_acc;
    }
    // grad_agents_ = grad_Q_A · W_q_agents + grad_K_A · W_k_agents
    for (size_t a = 0; a < num_agents_; ++a)
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j)
                acc += grad_Q_A(a, j) * W_q_agents.weights(j, i)
                     + grad_K_A(a, j) * W_k_agents.weights(j, i);
            grad_agents_(a, i) = acc;
        }

    // Step 7: grad_input = grad_Q · W_q + grad_K · W_k + grad_V · W_v
    // (Three Dense backward calls; each computes grad_input = grad_output @ W.)
    Tensor grad_input(N, d_model_);
    grad_input.fill(0.0);
    for (size_t t = 0; t < N; ++t)
        for (size_t i = 0; i < d_model_; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < d_model_; ++j)
                acc += grad_Q(t, j) * W_q.weights(j, i)
                     + grad_K(t, j) * W_k.weights(j, i)
                     + grad_V(t, j) * W_v.weights(j, i);
            grad_input(t, i) = acc;
        }

    return grad_input;
}

void AgentAttention::update_weights(double learning_rate) {
    W_q.update_weights(learning_rate);
    W_k.update_weights(learning_rate);
    W_v.update_weights(learning_rate);
    W_o.update_weights(learning_rate);
    W_q_agents.update_weights(learning_rate);
    W_k_agents.update_weights(learning_rate);
    // Agent tokens: SGD update
    for (size_t i = 0; i < num_agents_; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            agents_(i, j) -= learning_rate * grad_agents_(i, j);
    // log_lambda: SGD update
    log_lambda_(0, 0) -= learning_rate * grad_log_lambda_(0, 0);
}

void AgentAttention::zero_grad() {
    W_q.zero_grad();
    W_k.zero_grad();
    W_v.zero_grad();
    W_o.zero_grad();
    W_q_agents.zero_grad();
    W_k_agents.zero_grad();
    grad_agents_.fill(0.0);
    grad_log_lambda_.fill(0.0);
}

std::vector<Tensor*> AgentAttention::parameters() {
    std::vector<Tensor*> p = {
        &W_q.weights, &W_q.bias,
        &W_k.weights, &W_k.bias,
        &W_v.weights, &W_v.bias,
        &W_o.weights, &W_o.bias,
        &W_q_agents.weights, &W_q_agents.bias,
        &W_k_agents.weights, &W_k_agents.bias,
        &agents_, &log_lambda_
    };
    return p;
}

std::vector<Tensor*> AgentAttention::gradients() {
    std::vector<Tensor*> g = {
        &W_q.grad_weights, &W_q.grad_bias,
        &W_k.grad_weights, &W_k.grad_bias,
        &W_v.grad_weights, &W_v.grad_bias,
        &W_o.grad_weights, &W_o.grad_bias,
        &W_q_agents.grad_weights, &W_q_agents.grad_bias,
        &W_k_agents.grad_weights, &W_k_agents.grad_bias,
        &grad_agents_, &grad_log_lambda_
    };
    return g;
}

// ============================================================================
// AgentAttentionBlock
// ============================================================================

AgentAttentionBlock::AgentAttentionBlock(size_t d_model, size_t num_agents,
                                         bool use_linear_residual, size_t ffn_mult)
    : d_model_(d_model),
      num_agents_(num_agents),
      use_linear_residual_(use_linear_residual),
      attn(d_model, num_agents, use_linear_residual),
      ln1(d_model), ln2(d_model),
      ffn1(d_model, d_model * ffn_mult),
      ffn2(d_model * ffn_mult, d_model)
{}

Tensor AgentAttentionBlock::forward(const Tensor& input) {
    if (input.cols != d_model_) {
        throw std::invalid_argument("AgentAttentionBlock: input cols mismatch");
    }
    last_x_ = input.clone();

    // Pre-LN → attn → residual
    last_ln1_out_ = ln1.forward(input);
    Tensor attn_out = attn.forward(last_ln1_out_);
    last_attn_out_ = attn_out.clone();
    last_resid1_ = Tensor(input.rows, d_model_);
    for (size_t t = 0; t < input.rows; ++t)
        for (size_t j = 0; j < d_model_; ++j)
            last_resid1_(t, j) = input(t, j) + attn_out(t, j);

    // Pre-LN → FFN → residual
    last_ln2_out_ = ln2.forward(last_resid1_);
    Tensor h = ffn1.forward(last_ln2_out_);
    last_ffn_pregelu_ = h.clone();
    // GELU in-place
    for (size_t i = 0; i < h.rows; ++i)
        for (size_t j = 0; j < h.cols; ++j) {
            double x = h(i, j);
            h(i, j) = 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
        }
    Tensor h2 = ffn2.forward(h);
    last_ffn_out_ = h2.clone();
    (void)last_x_;  // suppress unused-vars warning (used in backward via stored input)
    Tensor output = Tensor(input.rows, d_model_);
    for (size_t t = 0; t < input.rows; ++t)
        for (size_t j = 0; j < d_model_; ++j)
            output(t, j) = last_resid1_(t, j) + h2(t, j);
    return output;
}

Tensor AgentAttentionBlock::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t N = grad_output.rows;
    if (grad_output.cols != d_model_) {
        throw std::invalid_argument("AgentAttentionBlock::backward: shape mismatch");
    }

    // grad to ffn2 output
    Tensor grad_h2 = grad_output.clone();
    // grad to ffn2 input (post-GELU)
    Tensor grad_h_postgelu = ffn2.backward(grad_h2, 0.0);
    // Apply GELU derivative
    for (size_t i = 0; i < grad_h_postgelu.rows; ++i)
        for (size_t j = 0; j < grad_h_postgelu.cols; ++j) {
            double x = last_ffn_pregelu_(i, j);
            double cdf = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
            double pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
            double gelu_d = cdf + x * pdf;
            grad_h_postgelu(i, j) *= gelu_d;
        }
    // grad to ffn1 input (LN output)
    Tensor grad_ln2 = ffn1.backward(grad_h_postgelu, 0.0);
    // grad to LN2 input (= resid1)
    Tensor grad_resid1 = ln2.backward(grad_ln2, 0.0);
    // add residual
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_resid1(i, j) += grad_output(i, j);

    // grad to attn out (= resid1 - input)
    Tensor grad_attn_out = grad_resid1.clone();
    // grad to attn input (LN1 output)
    Tensor grad_ln1 = attn.backward(grad_attn_out, 0.0);
    // grad to LN1 input
    Tensor grad_input = ln1.backward(grad_ln1, 0.0);
    // add residual
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < d_model_; ++j)
            grad_input(i, j) += grad_resid1(i, j);
    return grad_input;
}

void AgentAttentionBlock::update_weights(double learning_rate) {
    attn.update_weights(learning_rate);
    ln1.update_weights(learning_rate);
    ln2.update_weights(learning_rate);
    ffn1.update_weights(learning_rate);
    ffn2.update_weights(learning_rate);
}

void AgentAttentionBlock::zero_grad() {
    attn.zero_grad();
    ln1.zero_grad();
    ln2.zero_grad();
    ffn1.zero_grad();
    ffn2.zero_grad();
}

std::vector<Tensor*> AgentAttentionBlock::parameters() {
    auto p = attn.parameters();
    auto ln1p = ln1.parameters();
    auto ln2p = ln2.parameters();
    auto ffn1p = ffn1.parameters();
    auto ffn2p = ffn2.parameters();
    p.insert(p.end(), ln1p.begin(), ln1p.end());
    p.insert(p.end(), ln2p.begin(), ln2p.end());
    p.insert(p.end(), ffn1p.begin(), ffn1p.end());
    p.insert(p.end(), ffn2p.begin(), ffn2p.end());
    return p;
}

std::vector<Tensor*> AgentAttentionBlock::gradients() {
    auto g = attn.gradients();
    auto ln1g = ln1.gradients();
    auto ln2g = ln2.gradients();
    auto ffn1g = ffn1.gradients();
    auto ffn2g = ffn2.gradients();
    g.insert(g.end(), ln1g.begin(), ln1g.end());
    g.insert(g.end(), ln2g.begin(), ln2g.end());
    g.insert(g.end(), ffn1g.begin(), ffn1g.end());
    g.insert(g.end(), ffn2g.begin(), ffn2g.end());
    return g;
}

// ============================================================================
// AgentAttentionModel
// ============================================================================

AgentAttentionModel::AgentAttentionModel(size_t in_dim, size_t d_model, size_t out_features,
                                         size_t num_blocks, size_t num_agents,
                                         bool use_linear_residual, size_t ffn_mult)
    : d_model_(d_model),
      num_blocks_(num_blocks),
      out_features_(out_features),
      blocks_(num_blocks, AgentAttentionBlock(d_model, num_agents, use_linear_residual, ffn_mult)),
      final_ln_(d_model),
      input_proj_(in_dim, d_model),
      classifier_(d_model, out_features)
{
    if (in_dim == 0 || d_model == 0 || out_features == 0 || num_blocks == 0) {
        throw std::invalid_argument("AgentAttentionModel: dims and num_blocks must be > 0");
    }
}

Tensor AgentAttentionModel::forward(const Tensor& input) {
    if (input.cols != input_proj_.weights.cols) {
        throw std::invalid_argument("AgentAttentionModel::forward: input cols mismatch");
    }
    last_input_ = input.clone();
    last_in_proj_ = input_proj_.forward(input);
    Tensor x = last_in_proj_.clone();
    for (size_t i = 0; i < blocks_.size(); ++i) {
        x = blocks_[i].forward(x);
    }
    last_final_ln_ = final_ln_.forward(x);
    last_logits_ = classifier_.forward(last_final_ln_);
    return last_logits_;
}

Tensor AgentAttentionModel::backward(const Tensor& grad_output, double /*learning_rate*/) {
    Tensor grad = classifier_.backward(grad_output, 0.0);
    grad = final_ln_.backward(grad, 0.0);
    for (size_t i = blocks_.size(); i-- > 0;) {
        grad = blocks_[i].backward(grad, 0.0);
    }
    Tensor grad_in = input_proj_.backward(grad, 0.0);
    return grad_in;
}

void AgentAttentionModel::update_weights(double learning_rate) {
    for (auto& b : blocks_) b.update_weights(learning_rate);
    final_ln_.update_weights(learning_rate);
    input_proj_.update_weights(learning_rate);
    classifier_.update_weights(learning_rate);
}

void AgentAttentionModel::zero_grad() {
    for (auto& b : blocks_) b.zero_grad();
    final_ln_.zero_grad();
    input_proj_.zero_grad();
    classifier_.zero_grad();
}

std::vector<Tensor*> AgentAttentionModel::parameters() {
    std::vector<Tensor*> p;
    p = input_proj_.parameters();
    for (auto& b : blocks_) {
        auto bp = b.parameters();
        p.insert(p.end(), bp.begin(), bp.end());
    }
    auto lp = final_ln_.parameters();
    p.insert(p.end(), lp.begin(), lp.end());
    auto cp = classifier_.parameters();
    p.insert(p.end(), cp.begin(), cp.end());
    return p;
}

std::vector<Tensor*> AgentAttentionModel::gradients() {
    std::vector<Tensor*> g;
    g = input_proj_.gradients();
    for (auto& b : blocks_) {
        auto bg = b.gradients();
        g.insert(g.end(), bg.begin(), bg.end());
    }
    auto lg = final_ln_.gradients();
    g.insert(g.end(), lg.begin(), lg.end());
    auto cg = classifier_.gradients();
    g.insert(g.end(), cg.begin(), cg.end());
    return g;
}
