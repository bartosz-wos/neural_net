#ifndef NGPT_H
#define NGPT_H

#include "../../core/layer.h"
#include <vector>
#include <memory>
#include <cmath>

// ============================================================================
// nGPT — Normalized Transformer with Representation Learning on the Hypersphere
//   Loshchilov, Hsieh, Sun, Ginsburg (NVIDIA, 2024)
//   https://arxiv.org/abs/2410.01131
// ============================================================================
//
// Everything lives on the unit hypersphere. Let Norm(v) = v / ||v||_2. The paper
// constrains:
//   * every ROW of every projection matrix to unit norm,
//   * the residual-stream hidden state to unit norm after every update.
//
// LayerNorm / RMSNorm are removed ENTIRELY — normalization is a hard constraint
// on the parameters and the state, not a learned re-scaling applied at runtime.
// Because all vectors are unit-norm, a matmul is a vector of cosine similarities
// in [-1, 1], so the usual 1/sqrt(d) variance scaling is replaced by explicit
// learnable scalars (s below, and s_qk for the attention logits).
//
// The paper reports 4-20x fewer training steps to a given accuracy.
//
// ----------------------------------------------------------------------------
// HypersphereLinear
//
// Parameter W in R^(out, in) (same layout as Dense::weights) is stored
// UNNORMALIZED; the forward normalizes each row on the fly:
//
//   W_hat[o, :] = W[o, :] / ||W[o, :]||_2          (norm over the `in` axis)
//   y[t, o]     = s[o] * sum_c x[t, c] * W_hat[o, c]
//
// s in R^(1, out) is the paper's learnable per-output scale (the s_u / s_v / s_qk
// family, §2.3), init 1.
//
// Backward — the part that is easy to get wrong. For a row w with w_hat = w/||w||,
//
//   d w_hat_i / d w_j = (delta_ij - w_hat_i * w_hat_j) / ||w||
//
// so  dw = (I - w_hat w_hat^T) dw_hat / ||w|| = (dw_hat - w_hat (w_hat . dw_hat)) / ||w||.
// The projector (I - w_hat w_hat^T) kills the RADIAL component, i.e. a gradient
// step never changes ||w|| to first order — which is precisely why the paper can
// re-normalize after every optimizer step without fighting the optimizer.
//
// Per output row o:
//   dW_hat[o, c] = sum_t dy[t, o] * s[o] * x[t, c]
//   n_o          = ||W[o, :]||
//   r_o          = sum_c W_hat[o, c] * dW_hat[o, c]                 (radial part)
//   dW[o, c]     = (dW_hat[o, c] - W_hat[o, c] * r_o) / n_o
//   ds[o]        = sum_t dy[t, o] * (sum_c x[t, c] * W_hat[o, c])
//   dx[t, c]     = sum_o dy[t, o] * s[o] * W_hat[o, c]
//
// update_weights() performs the SGD step AND THEN re-normalizes every row —
// re-normalization is part of the step (paper §2.6), not a one-off at init.
//
// ----------------------------------------------------------------------------
// Normalized-LERP residual with eigen learning rates (paper Eq. 10/11)
//
// The standard pre-LN residual h <- h + f(LN(h)) becomes
//
//   h <- Norm( h + alpha * (Norm(f(h)) - h) ),      alpha in R^(1, d_model)
//
// alpha are the EIGEN LEARNING RATES: the diagonal of a learnable variable-metric
// matrix under the paper's Riemannian-optimization reading. alpha = 0 recovers the
// identity (h stays put); alpha = 1 recovers full replacement h <- Norm(f(h)).
// Because the update is a LERP toward a unit vector followed by a projection back
// onto the sphere, the state cannot blow up — the paper's stability claim.
//
// Backward, with u = Norm(f(h)), m = h + alpha * (u - h), h_out = Norm(m):
//   dm  = (dh_out - h_out (h_out . dh_out)) / ||m||         <- Norm Jacobian
//   dalpha = dm * (u - h)
//   du     = dm * alpha
//   dh    += dm * (1 - alpha)                               <- skip path
//   df     = (du - u (u . du)) / ||f(h)||                    <- Norm Jacobian
// Three applications of the same (I - v_hat v_hat^T)/||v|| Jacobian, implemented
// ONCE as `ngpt_norm_backward` and reused.
//
// ----------------------------------------------------------------------------
// QK scaling (paper §2.3)
//
// After HypersphereLinear the per-head q and k slices are built from unit-norm
// rows, so the logits are bounded and a bare softmax would be far too cold. The
// paper introduces a learnable s_qk and uses
//
//   scores[t, s] = (q_t . k_s) * s_qk * sqrt(head_dim)
//
// s_qk is a scalar Tensor(1, 1), init 1. Attention is strictly causal (s <= t).
//
// ----------------------------------------------------------------------------
// Deliberate deviations from the paper (see docs/plans/2026-09-08-ngpt-hypersphere.md)
//   1. MLP is a 2-layer GELU MLP rather than SwiGLU — matches every other block in
//      this repo, and SwiGLU is orthogonal to the hypersphere claim under test.
//   2. alpha init is a plain 0.05 rather than 0.05/sqrt(d_model): d_model in the
//      tests is 4-8, and the 1/sqrt(d) factor would put alpha at the FD noise floor.
//   3. NGPTModel's input projection is itself a HypersphereLinear, which is the
//      projection-based equivalent of the paper's normalized embedding table.
//
// Conventions (match fox.h / stick_breaking.h):
//   * Input / output: (N, d_model), N tokens, no explicit batch dim.
//   * Raw Tensor parameters with manually-accumulated grad_* (no Dense::backward).
//   * parameters()/gradients() return shape-matched pairs in the same order.
// ============================================================================

// Shared Norm-Jacobian helper: given v_hat = v/||v||, its norm, and dL/dv_hat,
// returns dL/dv = (dv_hat - v_hat * (v_hat . dv_hat)) / ||v||, row-wise.
// Rows with a (near-)zero norm are passed through as zero gradient.
Tensor ngpt_norm_backward(const Tensor& v_hat, const std::vector<double>& norms,
                          const Tensor& d_v_hat);

// Row-wise L2 normalization; also writes the per-row norms into `norms`.
Tensor ngpt_row_normalize(const Tensor& v, std::vector<double>& norms);

// ----------------------------------------------------------------------------
// HypersphereLinear — y = s * (x @ Norm_rows(W)^T)
// ----------------------------------------------------------------------------
class HypersphereLinear : public Layer {
public:
    HypersphereLinear(size_t in_features, size_t out_features);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W; }
    Tensor get_gradients() const override { return grad_W; }
    std::string name() const override { return "HypersphereLinear"; }

    size_t in_features()  const { return in_; }
    size_t out_features() const { return out_; }

    // Row-normalized view of W — the matrix the forward actually applies.
    Tensor normalized_weights() const;

    // Re-project every row of W onto the unit sphere (called by update_weights).
    void renormalize();

    Tensor W;        // (out, in) — stored unnormalized
    Tensor s;        // (1, out)  — learnable per-output scale
    Tensor grad_W;   // (out, in)
    Tensor grad_s;   // (1, out)

private:
    size_t in_, out_;
    Tensor last_input_;    // (N, in)
    Tensor last_W_hat_;    // (out, in)
    Tensor last_pre_s_;    // (N, out) — x @ W_hat^T, before the s scaling
    std::vector<double> last_norms_;   // per-row ||W[o, :]||
};

// ----------------------------------------------------------------------------
// NGPTBlock — causal attention sublayer + GELU MLP sublayer, each wrapped in the
// normalized-LERP residual. No LayerNorm anywhere.
// ----------------------------------------------------------------------------
class NGPTBlock : public Layer {
public:
    NGPTBlock(size_t d_model, size_t num_heads = 1, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.W; }
    Tensor get_gradients() const override { return W_q.grad_W; }
    std::string name() const override { return "NGPTBlock"; }

    size_t d_model()   const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim()  const { return head_dim_; }
    size_t ffn_dim()   const { return ffn_dim_; }

    // Norm(f_mlp(.)) from the last forward — the target the MLP LERP interpolates
    // toward. With alpha_mlp == 1 the block output equals this exactly.
    const Tensor& last_mlp_normalized() const { return last_u_mlp_; }
    // Attention probabilities from the last forward, (num_heads * N, N).
    const Tensor& last_attn() const { return last_attn_; }

    // Sublayer projections
    HypersphereLinear W_q, W_k, W_v, W_o;
    HypersphereLinear fc1, fc2;
    // nGPT-specific learnable scalars
    Tensor s_qk;          // (1, 1)   attention-logit scale
    Tensor alpha_attn;    // (1, d_model) eigen learning rates, attention sublayer
    Tensor alpha_mlp;     // (1, d_model) eigen learning rates, MLP sublayer
    Tensor grad_s_qk, grad_alpha_attn, grad_alpha_mlp;

private:
    size_t d_model_, num_heads_, head_dim_, ffn_dim_;
    double sqrt_dh_;

    // Attention sublayer caches
    Tensor last_input_;      // (N, d_model) = h_in
    Tensor last_Q_, last_K_, last_V_;
    Tensor last_attn_;       // (num_heads * N, N)
    Tensor last_head_out_;   // (N, d_model) pre-W_o
    Tensor last_f_attn_;     // (N, d_model) raw sublayer output
    Tensor last_u_attn_;     // (N, d_model) Norm(f_attn)
    std::vector<double> last_n_f_attn_;
    Tensor last_m_attn_;     // (N, d_model) LERP result pre-Norm
    std::vector<double> last_n_m_attn_;
    Tensor last_h_mid_;      // (N, d_model) Norm(m_attn) — attention sublayer output

    // MLP sublayer caches
    Tensor last_fc1_pre_;    // (N, ffn_dim) pre-GELU
    Tensor last_fc1_act_;    // (N, ffn_dim) post-GELU
    Tensor last_f_mlp_;
    Tensor last_u_mlp_;
    std::vector<double> last_n_f_mlp_;
    Tensor last_m_mlp_;
    std::vector<double> last_n_m_mlp_;
    Tensor last_h_out_;
};

// ----------------------------------------------------------------------------
// NGPTModel — HypersphereLinear input proj -> N x NGPTBlock -> Dense classifier
//
// The classifier is an ordinary unnormalized Dense so the output can leave the
// sphere; otherwise every regression target of norm != 1 would be unreachable.
// ----------------------------------------------------------------------------
class NGPTModel : public Layer {
public:
    NGPTModel(size_t input_dim, size_t d_model, size_t output_dim,
              size_t num_blocks, size_t num_heads = 1, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return classifier.weights; }
    Tensor get_gradients() const override { return classifier.grad_weights; }
    std::string name() const override { return "NGPTModel"; }

    size_t num_blocks() const { return blocks_.size(); }

    HypersphereLinear input_proj;
    std::vector<std::unique_ptr<NGPTBlock>> blocks_;
    Dense classifier;

private:
    size_t input_dim_, d_model_, output_dim_;
    // Cache for the pre-block row normalization (its Jacobian is needed in backward).
    Tensor last_embed_norm_;                // Norm_rows(input_proj(x))
    std::vector<double> last_embed_norms_;  // per-row ||input_proj(x)[t, :]||
};

#endif // NGPT_H
