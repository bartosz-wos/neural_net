# Agent Attention Implementation Plan (Han et al. 2024)

> **For Hermes:** Use test-driven-development skill. Build test-first.

**Goal:** Implement Agent Attention — a hybrid between softmax and linear attention that achieves softmax-quality attention at linear O(N) cost — in `include/nn/layers/attention/agent_attention.{h,cpp}`. Complements the existing `lsh_attention`, `linformer`, `performer`, `nystrom_attention`, `aft`, `gqa`, `mla`, and other efficient attention mechanisms in the repo.

**Architecture:** A single-head attention layer that introduces a small set of `N_a` "agent" tokens A (`N_a << N`). Two-stage attention:
1. Agent aggregation: `A' = softmax(Q_A · K^T) · V` (N_a × d) — agents gather info from all tokens.
2. Agent broadcast: `O = softmax(Q · K_A^T) · A'` (N × d) — tokens read from agents.
Optional: `O_final = O + λ · O_linear` for a residual linear-attention path.

**Tech Stack:** C++17, Tensor class, existing Layer patterns. Dense projections for Q/K/V/O + learnable agent tokens. Follows the patterns set by `performer.h`, `linformer.h`, `aft.h`.

---

## Paper reference

Han, Ye, Han, Xia, Pan, Wan, Song, Huang. **"Agent Attention: On the Integration of Softmax and Linear Attention."** ECCV 2024. https://arxiv.org/abs/2312.08874

## Core idea

Standard softmax attention is O(N²):
```
O = softmax(Q K^T / sqrt(d)) V
```

Linear attention (Performer, cosFormer) is O(N):
```
O = φ(Q) (φ(K)^T V) / (φ(Q) (φ(K)^T 1))
```
but suffers from a quality gap — the unordered feature map loses the sharpness of softmax.

**Agent Attention** sits between them: introduce N_a learnable **agent tokens** A ∈ R^(N_a × d). Each agent first aggregates info from all tokens (cheaper because N_a << N), then each token queries the agents:
```
A' = softmax(Q_A K^T / sqrt(d)) V        # (N_a, d)  — agent aggregation
O  = softmax(Q K_A^T / sqrt(d)) A'       # (N, d)    — agent broadcast
```
This is O(N · N_a · d) — linear when N_a is fixed. The paper optionally adds a residual linear attention path:
```
O_final = O + λ · φ(Q)(φ(K)^T V) / (φ(Q)(φ(K)^T 1))
```
Quality: matches softmax attention while being linear-time.

## Public API

```cpp
class AgentAttention : public Layer {
public:
    AgentAttention(size_t d_model, size_t seq_len, size_t num_agents,
                   bool use_linear_residual = true, double scale = 0.0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return W_q.grad_weights; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "AgentAttention"; }

    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    size_t num_agents() const { return num_agents_; }
    bool use_linear_residual() const { return use_linear_residual_; }
    double lambda() const { return lambda_; }
    const Tensor& agents() const { return agents_; }

    // Public params (for tests)
    Dense W_q;            // (d_model, d_model)
    Dense W_k;            // (d_model, d_model)
    Dense W_v;            // (d_model, d_model)
    Dense W_o;            // (d_model, d_model)
    Dense W_q_agents;     // (d_model, d_model) — query projection for the agent path
    Dense W_k_agents;     // (d_model, d_model) — key projection for agents
    Dense agent_proj;     // (d_model, d_model) — projects agents → K-space for K_A

    // Learnable agent tokens A ∈ R^(N_a, d_model)
    Tensor agents_;
    Tensor grad_agents_;
};

class AgentAttentionBlock : public Layer {
    // pre-LN → AgentAttention → residual → pre-LN → FFN → residual
};

class AgentAttentionModel : public Layer {
    // stack of AgentAttentionBlocks + final classifier
};
```

## Math

### Forward (eager / non-causal for general purpose)

Input: `x ∈ R^(N, d_model)`.

1. **Project**: `Q = x W_q^T`, `K = x W_k^T`, `V = x W_v^T` — each `(N, d_model)`.
2. **Agent K/Q**: `Q_A = agent_proj(A)` — `(N_a, d_model)` (treat A as a sequence of length N_a with its own projection).
3. **Agent K for broadcast**: `K_A = A W_k_agents^T` — `(N_a, d_model)`.
   (Following the paper: agents have their own K-V learned independently. Q_A is for the agent's query of all tokens; K_A is the agent's "key" for tokens to query them.)
4. **Agent aggregation**: `A' = softmax_row(Q_A K^T / sqrt(d)) V` — `(N_a, d_model)`.
   - `S_agg = Q_A K^T / sqrt(d)` — `(N_a, N)`.
   - `P_agg = row_softmax(P_agg)` — `(N_a, N)`.
   - `A' = P_agg V` — `(N_a, d)`.
5. **Agent broadcast**: `O = softmax_row(Q K_A^T / sqrt(d)) A'` — `(N, d_model)`.
   - `S_brd = Q K_A^T / sqrt(d)` — `(N, N_a)`.
   - `P_brd = row_softmax(S_brd)` — `(N, N_a)`.
   - `O = P_brd A'` — `(N, d)`.
6. **Linear residual** (optional, the paper's "DABA" variant):
   - `O_lin = (Q_mean * K^T) V` — but per-element, not aggregated. Concretely:
     - `Z_lin = φ(Q) (φ(K)^T V) / (φ(Q) (φ(K)^T 1))` — use softmax(QK^T)V directly without denominator for simplicity (the paper does this; the λ scaling absorbs the difference).
   - `O_final = O + λ · O_lin` where `λ` is a learnable scalar.
7. **Output**: `out = O_final W_o^T + b_o` — `(N, d_model)`.

We use **full softmax** for both stages (unlike the paper which sometimes uses lightweight features). For gradient checks we want exact attention.

### Backward (single-pass)

For each of the two stages, the backward is standard softmax-attention backward (closed-form `dS = P ⊙ (dO - sum)`, `dQ = dS · K`, `dK = dS^T · Q`, `dV = P^T · dO`).

**Stage 1 backward** (aggregation: receive `grad_A'` from stage 2):
- `dP_agg = grad_A' V^T`... wait, dimension-wise:
  - `A' = P_agg V`, P_agg is `(N_a, N)`, V is `(N, d)`, A' is `(N_a, d)`.
  - `grad_P_agg = grad_A' V^T` — `(N_a, N)`.
  - `grad_V = P_agg^T grad_A'` — `(N, d)` (accumulates across stages).
- Softmax backward: `grad_S_agg = P_agg ⊙ (grad_P_agg - rowsum(grad_P_agg ⊙ P_agg))`.
- `grad_Q_A = grad_S_agg K / sqrt(d)` — `(N_a, d)`.
- `grad_K = Q_A^T grad_S_agg / sqrt(d)` — `(N, d)`.
- `grad_agent_proj = grad_Q_A^T flattened for matmul` — handle via standard Dense backward through the agent projection step.

**Stage 2 backward** (broadcast: receive `grad_O` from the residual + linear residual):
- `grad_O_pre = grad_O · (1 + λ)` (if linear residual present) — back-propagated through the `O_final = O + λ · O_lin` add.
- `grad_P_brd = grad_O_pre A'^T` — `(N, N_a)`.
- `grad_A' += P_brd^T grad_O_pre` — `(N_a, d)` (accumulates with stage 1 grad above).
- Softmax backward: `grad_S_brd = P_brd ⊙ (grad_P_brd - rowsum(grad_P_brd ⊙ P_brd))`.
- `grad_Q = grad_S_brd K_A / sqrt(d)` — `(N, d)`.
- `grad_K_A = Q^T grad_S_brd / sqrt(d)` — `(N_a, d)`.

**Linear residual backward** (PADE-style):
- `O_lin = softmax(QK^T / sqrt(d)) V` (full softmax attention, not the more efficient linear-attention form — the paper allows both).
- Standard softmax attention backward.

**Agent token gradient** (the trickiest part):
- `A` is a parameter. It flows through:
  1. `Q_A = A_agent_proj(A)` (or `A · W_q_agents^T`).
  2. `K_A = A · W_k_agents^T`.
- `grad_A` accumulates from:
  - `grad_A` from `grad_Q_A` via `W_q_agents^T` backward.
  - `grad_A` from `grad_K_A` via `W_k_agents^T` backward.

**W_o backward**: standard Dense backward on `grad_O_final → W_o`.
**W_q/W_k/W_v backward**: standard Dense backward on `grad_Q/grad_K/grad_V`.

## What this adds vs. existing layers

- **vs. Performer**: random feature approximation (FAVOR+) → lossy. Agent Attention uses exact softmax with two stages. Same complexity but higher quality.
- **vs. Linformer**: low-rank approximation of K^T V → fixed projection. Agent Attention has learnable agents that adapt to the input.
- **vs. Nystrom**: landmark-based low-rank → similar in spirit but uses Nystrom formula (not softmax). Agent Attention uses softmax agents.
- **vs. LSH**: bucketing discards pairs → lossy. Agent Attention keeps all-pair information via the two-stage softmax.

## Files

- `include/nn/layers/attention/agent_attention.h` — new
- `include/nn/layers/attention/agent_attention.cpp` — new
- `tests/test_agent_attention.cpp` — new
- `include/nn/nn.h` — add `#include "layers/attention/agent_attention.h"` (in the attention section)
- `Makefile` — add `build/test_agent_attention` rule + `tests:` deps entry + `=== Running Agent Attention Tests ===` echo in `run_tests`
- `docs/plans/2026-08-21-agent-attention.md` — this plan

## Tests (target ~15 focused checks)

1. **Constructor validation** (5 cases): d_model=0, seq_len=0, num_agents=0, num_agents > seq_len, scale invalid.
2. **Forward shape**: (N=4, d_model=4) → (N=4, d_model=4), with N_a=2.
3. **Forward shape with linear residual**: same shape, params include λ.
4. **Forward finiteness + nonzero** for N=4, N_a=2.
5. **Forward shape with N_a=1** (degenerate but allowed).
6. **Agent gathering structurally correct**: A' is a weighted combination of V rows (each row of A' is a weighted sum of V rows).
7. **Broadcast structurally correct**: out_t is a weighted combination of A' rows (each row of out is a weighted sum of A' rows).
8. **Input gradient FD check** (rel_err < 1e-3).
9. **W_q gradient FD check**.
10. **W_k gradient FD check**.
11. **W_v gradient FD check**.
12. **W_o gradient FD check**.
13. **agent_proj gradient FD check** (exercises the agent → Q_A projection).
14. **Agents (A) gradient FD check** — the LEARNABLE parameter that's the centerpiece; non-zero gradient confirms the agents are being updated.
15. **Linear residual λ gradient FD check** (when present).
16. **Determinism**: two fresh AgentAttentions with copied params → bit-exact forward.
17. **Training reduces loss** (50 SGD steps, regression-style target).
18. **Param/grad count contract** (uniform across `use_linear_residual` flag).
19. **AgentAttentionModel forward shape + training reduces loss**.

Out of scope: multi-head, causal masking, FLOPs-optimized FFV path (the paper's "Agent Attention v2" extensions). Single-head eager for gradient check tractability.

## Pitfalls / TDD-isms from past sessions

- **Don't use random init for FD checks** (uniform init masks row-vs-column confusion). Use deterministic non-uniform init via `0.3 + 0.1 * (i + j) % 5`-style formulas. See `tests/test_linoss.cpp` for the pattern.
- **Use `2*(output - target)` for the loss gradient (mean-MSE convention)**, not `(output - target)/N` — the FD check uses `mse_loss = 0.5 * sum((output - target)^2)`, so the gradient w.r.t. output is `(output - target)`, not divided by N. See `tests/test_performer.cpp` for the pattern.
- **Mutation test**: stub out the agent gradient path (`grad_agents_.fill(0)` or skip the line) → test 14 should fail. Stub out the linear residual λ (`grad_lambda = 0` or multiply by 0) → test 15 should fail.
- **Verify the FD check has expected scale**: input gradient ~ 1e-9, parameter gradients ~ 1e-9 to 1e-11. If rel_err is 1e-2 or worse, the gradient chain is wrong.
- **AgentAttentionBlock's Model.add_layer** registration: make sure the test creates a `Model`, calls `model.add_layer(new AgentAttentionBlock(...))`, and that the optimizer actually iterates over the layer (the systematic-debugging 5d silent-no-op trap).

## Execution

1. Write `tests/test_agent_attention.cpp` (failing tests for the constructor + forward shape).
2. Write `include/nn/layers/attention/agent_attention.h` (declarations).
3. Write `include/nn/layers/attention/agent_attention.cpp` (implement enough to pass the constructor + forward tests).
4. Run `make build/test_agent_attention` and `./build/test_agent_attention` — confirm constructor + forward tests pass.
5. Extend the test file with FD gradient checks.
6. Extend the implementation with backward pass.
7. Run tests again — confirm gradient checks pass at machine precision.
8. Add `AgentAttentionBlock` and `AgentAttentionModel` and their tests.
9. Register in `include/nn/nn.h` and `Makefile`.
10. Run the full test suite to confirm no regressions.
11. Commit + push.
12. Move from `EXPANSION_QUEUE.md` `## Ideas` to `## Done`.
