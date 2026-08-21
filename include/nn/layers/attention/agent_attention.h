#ifndef AGENT_ATTENTION_H
#define AGENT_ATTENTION_H

#include "../../core/layer.h"
#include "../../core/model.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <memory>

// ============================================================================
// Agent Attention — Han et al. 2024 (ECCV 2024)
//   "Agent Attention: On the Integration of Softmax and Linear Attention"
//   https://arxiv.org/abs/2312.08874
//
// Innovation: a hybrid between softmax and linear attention that achieves
// softmax-quality attention at O(N · N_a) cost (linear in N when N_a is
// fixed). The key idea is to introduce a small set of N_a LEARNABLE "agent"
// tokens A ∈ R^(N_a × d_model) and split the attention into two stages:
//
//   Stage 1 (agent aggregation, N_a × N interactions):
//     A' = softmax(Q_A · K^T / sqrt(d)) · V          (N_a × d_model)
//
//   Stage 2 (agent broadcast, N × N_a interactions):
//     O  = softmax(Q   · K_A^T / sqrt(d)) · A'        (N × d_model)
//
// Where Q_A = A · W_q_agents^T and K_A = A · W_k_agents^T are the agent's
// own query/key projections (the agents have their own parameters, like
// extra tokens). The two-stage softmax reproduces the expressivity of full
// softmax attention while reducing complexity from O(N²) to O(N · N_a).
//
// Optional residual path (paper's "DABA" extension):
//   O_final = O + λ · O_linear
// where O_linear = softmax(Q · K^T / sqrt(d)) · V is a full softmax
// attention computation (the paper experiments with both linear and
// softmax forms; we use softmax for tractability and gradient clarity).
// λ is a learnable scalar initialized to 0.1.
//
// ----------------------------------------------------------------------------
// Why this is sub-quadratic
// ----------------------------------------------------------------------------
//   Standard softmax attention:   O(N² · d)    memory + compute
//   Agent attention:             O(N · N_a · d) when N_a is fixed
//   Performer / linear:          O(N · m · d)  where m = # features
//   For N_a = sqrt(N), agent attention matches softmax cost.
//   For N_a = constant, agent attention is linear in N.
//
// ----------------------------------------------------------------------------
// Shape conventions (matching Performer / Linformer / AFT in this repo)
// ----------------------------------------------------------------------------
//   (N, d_model) input/output — row-major
//   Single-head: for multi-head, call multiple AgentAttentions and
//   concatenate (same convention as LinformerAttention).
//   Pre-LN block pattern (pre-LN → attn → residual → pre-LN → FFN → residual).
//
// Classes:
//   AgentAttention       — single-head agent attention
//   AgentAttentionBlock  — pre-LN → AgentAttention → residual → pre-LN → FFN → residual
//   AgentAttentionModel  — stack of AgentAttentionBlocks + classifier
// ============================================================================

// ---------------------------------------------------------------------------
// AgentAttention — single-head agent attention
// ---------------------------------------------------------------------------
class AgentAttention : public Layer {
public:
    // d_model:    input/output feature dim
    // num_agents: number of learnable agent tokens (N_a)
    // use_linear_residual: if true, additionally compute full softmax attention
    //                      and add λ · O_linear to the output (paper's DABA).
    //                      O_final = O_agent + λ · O_linear.
    AgentAttention(size_t d_model, size_t num_agents,
                   bool use_linear_residual = true);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return W_q.grad_weights; }
    std::string name() const override { return "AgentAttention"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t num_agents() const { return num_agents_; }
    bool use_linear_residual() const { return use_linear_residual_; }

    // Public parameters (for tests + manual copying)
    Dense W_q;            // (d_model, d_model) — token → Q
    Dense W_k;            // (d_model, d_model) — token → K
    Dense W_v;            // (d_model, d_model) — token → V
    Dense W_o;            // (d_model, d_model) — output projection
    Dense W_q_agents;     // (d_model, d_model) — agent → Q_A
    Dense W_k_agents;     // (d_model, d_model) — agent → K_A

    // Learnable agent tokens A ∈ R^(N_a, d_model)
    Tensor agents_;       // (N_a, d_model)
    Tensor grad_agents_;  // (N_a, d_model)

    // Learnable residual scaling λ (scalar, (1, 1))
    Tensor log_lambda_;    // (1, 1) — λ = softplus(log_lambda_), always ≥ 0
    Tensor grad_log_lambda_;

    // BPTT cache (size N at runtime, but seq length is dynamic)
    Tensor last_input_;        // (N, d_model)
    Tensor last_q_;            // (N, d_model)
    Tensor last_k_;            // (N, d_model)
    Tensor last_v_;            // (N, d_model)
    Tensor last_qa_;           // (N_a, d_model) — agent query
    Tensor last_ka_;           // (N_a, d_model) — agent key
    Tensor last_A_prime_;      // (N_a, d_model) — agent output
    Tensor last_attn_agg_;     // (N_a, N) — softmax(Q_A K^T / sqrt(d))
    Tensor last_attn_brd_;     // (N, N_a) — softmax(Q K_A^T / sqrt(d))
    Tensor last_attn_lin_;     // (N, N) — softmax(Q K^T / sqrt(d)) (linear residual)
    Tensor last_O_agent_;      // (N, d_model) — pre-W_o agent output
    Tensor last_O_lin_;        // (N, d_model) — pre-add linear residual (zero if disabled)
    Tensor last_O_final_;      // (N, d_model) — post-add, pre-W_o
    double last_lambda_;       // scalar — λ at forward time

    size_t d_model_;
    size_t num_agents_;
    bool use_linear_residual_;

    // Cached for size N (= input.rows in forward)
    size_t N_last_;

private:
    // Helpers
    static double scaled_dot(double a, double b, double inv_sqrt_d);
};

// ---------------------------------------------------------------------------
// AgentAttentionBlock: pre-LN → AgentAttention → residual →
//                     pre-LN → FFN (2-layer GELU) → residual
//
// Mirrors the standard transformer block layout. The FFN is a 2-layer Dense
// with GELU activation (same as PerformerBlock / LinformerBlock).
// ---------------------------------------------------------------------------
class AgentAttentionBlock : public Layer {
public:
    size_t d_model_;
    size_t num_agents_;
    bool use_linear_residual_;

    AgentAttention attn;
    LayerNorm ln1, ln2;
    Dense ffn1;   // (d_model, ffn_dim)
    Dense ffn2;   // (ffn_dim, d_model)

    // BPTT cache
    Tensor last_x_;
    Tensor last_ln1_out_;
    Tensor last_attn_out_;
    Tensor last_resid1_;
    Tensor last_ln2_out_;
    Tensor last_ffn_pregelu_;
    Tensor last_ffn_out_;

    AgentAttentionBlock(size_t d_model, size_t num_agents,
                        bool use_linear_residual = true,
                        size_t ffn_mult = 4);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return ffn1.weights; }
    Tensor get_gradients() const override { return ffn1.grad_weights; }
    std::string name() const override { return "AgentAttentionBlock"; }
};

// ---------------------------------------------------------------------------
// AgentAttentionModel: stack of `num_blocks` AgentAttentionBlocks + classifier
// ---------------------------------------------------------------------------
class AgentAttentionModel : public Layer {
public:
    size_t d_model_;
    size_t num_blocks_;
    size_t out_features_;
    std::vector<AgentAttentionBlock> blocks_;
    LayerNorm final_ln_;
    Dense input_proj_;   // (in_dim, d_model)
    Dense classifier_;   // (d_model, out_features)

    // BPTT cache
    Tensor last_input_;
    Tensor last_in_proj_;
    Tensor last_final_ln_;
    Tensor last_logits_;

    AgentAttentionModel(size_t in_dim, size_t d_model, size_t out_features,
                        size_t num_blocks = 2, size_t num_agents = 4,
                        bool use_linear_residual = true, size_t ffn_mult = 4);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return input_proj_.weights; }
    Tensor get_gradients() const override { return input_proj_.grad_weights; }
    std::string name() const override { return "AgentAttentionModel"; }
};

#endif // AGENT_ATTENTION_H
