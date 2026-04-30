// TabNet: Hierarchical Attention for Tabular Deep Learning
// Arik & Pfister 2019 — https://arxiv.org/abs/1908.07442
#include "nn/layers/architectures/tabnet.h"
#include "nn/activations/activations.h"
#include <cmath>
#include <algorithm>

namespace {

void batch_norm_backward_1d(const Tensor& x, const Tensor& grad_out,
                            const Tensor& gamma,
                            const Tensor& mean, const Tensor& inv_std,
                            Tensor& grad_gamma, Tensor& grad_beta,
                            Tensor& grad_x) {
    size_t batch = x.rows;
    size_t features = x.cols;

    for (size_t f = 0; f < features; ++f) {
        double inv_s = inv_std[0][f];
        double mean_val = mean[0][f];

        double g_gamma = 0.0, g_beta = 0.0;
        for (size_t b = 0; b < batch; ++b) {
            double x_norm = (x[b][f] - mean_val) * inv_s;
            g_gamma += grad_out[b][f] * x_norm;
            g_beta  += grad_out[b][f];
        }
        grad_gamma[0][f] += g_gamma;
        grad_beta[0][f]  += g_beta;

        for (size_t b = 0; b < batch; ++b) {
            double x_norm = (x[b][f] - mean_val) * inv_s;
            grad_x[b][f] = inv_s * gamma[0][f] * grad_out[b][f]
                         - inv_s * gamma[0][f] * inv_s * inv_s * x_norm * g_gamma / batch
                         - inv_s * gamma[0][f] * g_beta / batch;
        }
    }
}

} // anonymous namespace

TabNet::TabNet(int input_dim, int num_outputs, int num_steps,
               int num_attention_heads, int n_independent,
               double relaxation_factor)
    : input_dim_(input_dim), num_outputs_(num_outputs),
      num_steps_(num_steps), num_attention_heads_(num_attention_heads),
      n_independent_(n_independent), relaxation_factor_(relaxation_factor),
      virtual_dim_(input_dim * 2), training_(false)
{
    // Shared encoder
    shared_encoder_.w1 = Tensor::random(virtual_dim_, input_dim_, 0.02);
    shared_encoder_.b1 = Tensor::zeros(1, virtual_dim_);
    shared_encoder_.bn1_gamma = Tensor::zeros(1, input_dim_); for(size_t __i=0;__i<(size_t)input_dim_;__i++) shared_encoder_.bn1_gamma[0][__i]=1.0;
    shared_encoder_.bn1_beta  = Tensor::zeros(1, input_dim_);
    shared_encoder_.last_bn1_out = Tensor(0, 0);
    shared_encoder_.w2 = Tensor::random(virtual_dim_, virtual_dim_, 0.02);
    shared_encoder_.b2 = Tensor::zeros(1, virtual_dim_);
    shared_encoder_.bn2_gamma = Tensor::zeros(1, virtual_dim_); for(size_t __i=0;__i<(size_t)virtual_dim_;__i++) shared_encoder_.bn2_gamma[0][__i]=1.0;
    shared_encoder_.bn2_beta  = Tensor::zeros(1, virtual_dim_);
    shared_encoder_.last_bn2_out = Tensor(0, 0);
    shared_encoder_.grad_bn1_gamma = Tensor::zeros(1, input_dim_);
    shared_encoder_.grad_bn1_beta  = Tensor::zeros(1, input_dim_);
    shared_encoder_.grad_bn2_gamma = Tensor::zeros(1, virtual_dim_);
    shared_encoder_.grad_bn2_beta  = Tensor::zeros(1, virtual_dim_);
    shared_encoder_.grad_w1 = Tensor::zeros(virtual_dim_, input_dim_);
    shared_encoder_.grad_b1 = Tensor::zeros(1, virtual_dim_);
    shared_encoder_.grad_w2 = Tensor::zeros(virtual_dim_, virtual_dim_);
    shared_encoder_.grad_b2 = Tensor::zeros(1, virtual_dim_);

    // Per-step blocks
    for (int s = 0; s < num_steps_; ++s) {
        IndependentBlock indep;
        indep.w1 = Tensor::random(virtual_dim_, virtual_dim_, 0.02);
        indep.b1 = Tensor::zeros(1, virtual_dim_);
        indep.bn1_gamma = Tensor::zeros(1, virtual_dim_); for(size_t __i=0;__i<(size_t)virtual_dim_;__i++) indep.bn1_gamma[0][__i]=1.0;
        indep.bn1_beta  = Tensor::zeros(1, virtual_dim_);
        indep.last_bn1_out = Tensor(0, 0);
        indep.w2 = Tensor::random(virtual_dim_, virtual_dim_, 0.02);
        indep.b2 = Tensor::zeros(1, virtual_dim_);
        indep.bn2_gamma = Tensor::zeros(1, virtual_dim_); for(size_t __i=0;__i<(size_t)virtual_dim_;__i++) indep.bn2_gamma[0][__i]=1.0;
        indep.bn2_beta  = Tensor::zeros(1, virtual_dim_);
        indep.last_bn2_out = Tensor(0, 0);
        indep.grad_bn1_gamma = Tensor::zeros(1, virtual_dim_);
        indep.grad_bn1_beta  = Tensor::zeros(1, virtual_dim_);
        indep.grad_bn2_gamma = Tensor::zeros(1, virtual_dim_);
        indep.grad_bn2_beta  = Tensor::zeros(1, virtual_dim_);
        indep.grad_w1 = Tensor::zeros(virtual_dim_, virtual_dim_);
        indep.grad_b1 = Tensor::zeros(1, virtual_dim_);
        indep.grad_w2 = Tensor::zeros(virtual_dim_, virtual_dim_);
        indep.grad_b2 = Tensor::zeros(1, virtual_dim_);
        independent_blocks_.push_back(indep);

        AttentionBlock attn;
        attn.w = Tensor::random(1, virtual_dim_, 0.02);
        attn.last_scores = Tensor(0, 0);
        attn.last_mask = Tensor(0, 0);
        attn.grad_w = Tensor::zeros(1, virtual_dim_);
        attention_blocks_.push_back(attn);

        StepFC fc;
        fc.w = Tensor::random(num_outputs_, input_dim_, 0.02);
        fc.b = Tensor::zeros(1, num_outputs_);
        fc.grad_w = Tensor::zeros(num_outputs_, input_dim_);
        fc.grad_b = Tensor::zeros(1, num_outputs_);
        step_fcs_.push_back(fc);

        {
            Tensor ps = Tensor::zeros(1, input_dim_);
            for(size_t __i=0;__i<(size_t)input_dim_;__i++) ps[0][__i]=1.0;
            prior_scales_.push_back(ps);
        }
    }

    // Build parameter list
    param_list_.push_back(&shared_encoder_.w1);
    param_list_.push_back(&shared_encoder_.b1);
    param_list_.push_back(&shared_encoder_.bn1_gamma);
    param_list_.push_back(&shared_encoder_.bn1_beta);
    param_list_.push_back(&shared_encoder_.w2);
    param_list_.push_back(&shared_encoder_.b2);
    param_list_.push_back(&shared_encoder_.bn2_gamma);
    param_list_.push_back(&shared_encoder_.bn2_beta);
    for (int s = 0; s < num_steps_; ++s) {
        param_list_.push_back(&independent_blocks_[s].w1);
        param_list_.push_back(&independent_blocks_[s].b1);
        param_list_.push_back(&independent_blocks_[s].bn1_gamma);
        param_list_.push_back(&independent_blocks_[s].bn1_beta);
        param_list_.push_back(&independent_blocks_[s].w2);
        param_list_.push_back(&independent_blocks_[s].b2);
        param_list_.push_back(&independent_blocks_[s].bn2_gamma);
        param_list_.push_back(&independent_blocks_[s].bn2_beta);
        param_list_.push_back(&attention_blocks_[s].w);
        param_list_.push_back(&step_fcs_[s].w);
        param_list_.push_back(&step_fcs_[s].b);
    }
    grad_list_.push_back(&shared_encoder_.grad_w1);
    grad_list_.push_back(&shared_encoder_.grad_b1);
    grad_list_.push_back(&shared_encoder_.grad_bn1_gamma);
    grad_list_.push_back(&shared_encoder_.grad_bn1_beta);
    grad_list_.push_back(&shared_encoder_.grad_w2);
    grad_list_.push_back(&shared_encoder_.grad_b2);
    grad_list_.push_back(&shared_encoder_.grad_bn2_gamma);
    grad_list_.push_back(&shared_encoder_.grad_bn2_beta);
    for (int s = 0; s < num_steps_; ++s) {
        grad_list_.push_back(&independent_blocks_[s].grad_w1);
        grad_list_.push_back(&independent_blocks_[s].grad_b1);
        grad_list_.push_back(&independent_blocks_[s].grad_bn1_gamma);
        grad_list_.push_back(&independent_blocks_[s].grad_bn1_beta);
        grad_list_.push_back(&independent_blocks_[s].grad_w2);
        grad_list_.push_back(&independent_blocks_[s].grad_b2);
        grad_list_.push_back(&independent_blocks_[s].grad_bn2_gamma);
        grad_list_.push_back(&independent_blocks_[s].grad_bn2_beta);
        grad_list_.push_back(&attention_blocks_[s].grad_w);
        grad_list_.push_back(&step_fcs_[s].grad_w);
        grad_list_.push_back(&step_fcs_[s].grad_b);
    }
}

Tensor TabNet::forward(const Tensor& input) {
    const size_t batch = input.rows;
    const size_t D = input_dim_;
    last_h_.clear();
    last_h_indep_.clear();
    last_masked_h_.clear();
    last_masks_.clear();

    // h starts at zeros (batch, num_outputs), accumulates per step
    Tensor h = Tensor::zeros(batch, num_outputs_);

    for (int s = 0; s < num_steps_; ++s) {
        Tensor h(batch, D);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < D; ++f)
                h[b][f] = input[b][f];
        last_h_.push_back(h);

        // Shared encoder
        EncoderBlock& se = shared_encoder_;
        Tensor bn1_out(batch, input_dim_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < D; ++f) {
                double x_norm = input[b][f];
                bn1_out[b][f] = se.bn1_gamma[0][f] * x_norm + se.bn1_beta[0][f];
            }
        se.last_bn1_out = bn1_out;

        Tensor h1(batch, virtual_dim_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t j = 0; j < (size_t)virtual_dim_; ++j) {
                double sum = se.b1[0][j];
                for (size_t i = 0; i < D; ++i)
                    sum += bn1_out[b][i] * se.w1[j][i];
                h1[b][j] = sum;
            }

        Tensor bn2_out(batch, virtual_dim_);
        double eps = 1e-5;
        Tensor mean(1, virtual_dim_), var(1, virtual_dim_), inv_std(1, virtual_dim_);
        mean.fill(0.0);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
                mean[0][f] += h1[b][f];
        for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
            mean[0][f] /= batch;
        var.fill(0.0);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f) {
                double d = h1[b][f] - mean[0][f];
                var[0][f] += d * d;
            }
        for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
            var[0][f] /= batch;
        for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
            inv_std[0][f] = 1.0 / std::sqrt(var[0][f] + eps);
        se.last_inv_std_bn2 = inv_std;  // store for backward
        se.last_mean_bn2 = mean;        // store for backward
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f) {
                double x_norm = (h1[b][f] - mean[0][f]) * inv_std[0][f];
                bn2_out[b][f] = se.bn2_gamma[0][f] * x_norm + se.bn2_beta[0][f];
            }
        se.last_bn2_out = bn2_out;

        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
                if (bn2_out[b][f] < 0) bn2_out[b][f] = 0.0;

        // Independent encoder
        IndependentBlock& indep = independent_blocks_[s];
        Tensor ibn1_out(batch, virtual_dim_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f) {
                double x_norm = bn2_out[b][f];
                ibn1_out[b][f] = indep.bn1_gamma[0][f] * x_norm + indep.bn1_beta[0][f];
            }
        indep.last_bn1_out = ibn1_out;

        Tensor h_indep1(batch, virtual_dim_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t j = 0; j < (size_t)virtual_dim_; ++j) {
                double sum = indep.b1[0][j];
                for (size_t i = 0; i < (size_t)virtual_dim_; ++i)
                    sum += ibn1_out[b][i] * indep.w1[j][i];
                h_indep1[b][j] = sum;
            }

        Tensor ibn2_out(batch, virtual_dim_);
        Tensor mean2(1, virtual_dim_), var2(1, virtual_dim_), inv_std2(1, virtual_dim_);
        mean2.fill(0.0);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
                mean2[0][f] += h_indep1[b][f];
        for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
            mean2[0][f] /= batch;
        var2.fill(0.0);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f) {
                double d = h_indep1[b][f] - mean2[0][f];
                var2[0][f] += d * d;
            }
        for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
            var2[0][f] /= batch;
        for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
            inv_std2[0][f] = 1.0 / std::sqrt(var2[0][f] + eps);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f) {
                double x_norm = (h_indep1[b][f] - mean2[0][f]) * inv_std2[0][f];
                ibn2_out[b][f] = indep.bn2_gamma[0][f] * x_norm + indep.bn2_beta[0][f];
            }
        indep.last_bn2_out = ibn2_out;

        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
                if (ibn2_out[b][f] < 0) ibn2_out[b][f] = 0.0;

        last_h_indep_.push_back(ibn2_out);

        // Attention
        AttentionBlock& attn = attention_blocks_[s];
        const Tensor& P = prior_scales_[s];

        Tensor scores(batch, virtual_dim_);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t vd = 0; vd < (size_t)virtual_dim_; ++vd) {
                double sum = attn.w[0][vd];
                for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
                    sum += ibn2_out[b][f] * (f == vd ? 1.0 : 0.0);
                scores[b][vd] = (P[0][vd % D] / relaxation_factor_) * sum;
            }
        }
        attn.last_scores = scores;

        Tensor exp_scores(batch, virtual_dim_);
        double max_score = -1e100;
        for (size_t b = 0; b < batch; ++b)
            for (size_t vd = 0; vd < (size_t)virtual_dim_; ++vd)
                if (scores[b][vd] > max_score) max_score = scores[b][vd];

        for (size_t b = 0; b < batch; ++b) {
            double sum_exp = 0.0;
            for (size_t vd = 0; vd < (size_t)virtual_dim_; ++vd) {
                exp_scores[b][vd] = std::exp(scores[b][vd] - max_score);
                sum_exp += exp_scores[b][vd];
            }
            for (size_t vd = 0; vd < (size_t)virtual_dim_; ++vd)
                exp_scores[b][vd] /= sum_exp;
        }
        attn.last_scores = exp_scores;

        Tensor mask(batch, D);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t f = 0; f < D; ++f) {
                double sum_attn = 0.0;
                for (size_t g = 0; g < (size_t)virtual_dim_; ++g)
                    if ((int)g % D == (int)f)
                        sum_attn += exp_scores[b][g];
                mask[b][f] = sum_attn;
            }
        }
        attn.last_mask = mask;
        last_masks_.push_back(mask);

        Tensor masked_h(batch, D);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < D; ++f)
                masked_h[b][f] = mask[b][f] * input[b][f];
        last_masked_h_.push_back(masked_h);

        StepFC& fc = step_fcs_[s];
        Tensor step_out(batch, num_outputs_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t j = 0; j < (size_t)num_outputs_; ++j) {
                double sum = fc.b[0][j];
                for (size_t i = 0; i < D; ++i)
                    sum += masked_h[b][i] * fc.w[j][i];
                step_out[b][j] = sum;
            }

        h += step_out;

        Tensor& P_new = prior_scales_[s];
        for (size_t f = 0; f < D; ++f)
            P_new[0][f] = std::max(0.0, P_new[0][f] - mask[0][f] / relaxation_factor_);
    }

    return h;
}

Tensor TabNet::backward(const Tensor& grad_output, double) {
    const size_t batch = grad_output.rows;
    const size_t D = input_dim_;
    Tensor grad_h = grad_output;

    // Accumulate gradients from all steps for the shared encoder
    Tensor grad_h_shared_accum = Tensor::zeros(batch, virtual_dim_);

    // Per-step backward
    for (int s = num_steps_ - 1; s >= 0; --s) {
        IndependentBlock& indep = independent_blocks_[s];
        AttentionBlock& attn = attention_blocks_[s];
        StepFC& fc = step_fcs_[s];
        const Tensor& mask = last_masks_[s];
        const Tensor& masked_h = last_masked_h_[s];
        const Tensor& h_indep = last_h_indep_[s];
        const Tensor& P = prior_scales_[s];

        // FC backward
        for (size_t j = 0; j < (size_t)num_outputs_; ++j)
            for (size_t i = 0; i < (size_t)input_dim_; ++i)
                for (size_t b = 0; b < batch; ++b)
                    fc.grad_w[j][i] += grad_h[b][j] * masked_h[b][i];
        for (size_t j = 0; j < (size_t)num_outputs_; ++j)
            for (size_t b = 0; b < batch; ++b)
                fc.grad_b[0][j] += grad_h[b][j];

        // dL/d_masked_h = dL/d_step * W^T
        Tensor grad_masked_h(batch, input_dim_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t i = 0; i < (size_t)input_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < (size_t)num_outputs_; ++j)
                    sum += grad_h[b][j] * fc.w[j][i];
                grad_masked_h[b][i] = sum;
            }

        // dL/d_input = mask * grad_masked_h (STE: mask gradient = 1)
        Tensor grad_input_step(batch, input_dim_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)input_dim_; ++f)
                grad_input_step[b][f] = mask[b][f] * grad_masked_h[b][f];

        // Attention grad_w (simplified)
        for (size_t vd = 0; vd < (size_t)virtual_dim_; ++vd) {
            double w_grad = 0.0;
            for (size_t b = 0; b < batch; ++b) {
                double h_val = 0.0;
                for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
                    h_val += h_indep[b][f] * (f == vd ? 1.0 : 0.0);
                for (size_t f = 0; f < (size_t)input_dim_; ++f)
                    w_grad += grad_masked_h[b][f] * (P[0][f] / relaxation_factor_) * h_val;
            }
            attn.grad_w[0][vd] = w_grad;
        }

        // Backward through independent encoder
        // BN2: dL/d_h_indep1 = dL/d_ibn2 * gamma2
        Tensor grad_ibn2(batch, virtual_dim_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
                grad_ibn2[b][f] = grad_masked_h[b][f % input_dim_];

        // BN2 backward
        Tensor mean2(1, virtual_dim_), inv_std2(1, virtual_dim_);
        mean2.fill(0.0);
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
                mean2[0][f] += indep.last_bn1_out[b][f];
        for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
            mean2[0][f] /= batch;
        for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
            inv_std2[0][f] = 1.0;

        Tensor grad_bn2_gamma = Tensor::zeros(1, virtual_dim_);
        Tensor grad_bn2_beta = Tensor::zeros(1, virtual_dim_);
        Tensor grad_bn2_in(batch, virtual_dim_);
        batch_norm_backward_1d(indep.last_bn1_out, grad_ibn2, indep.bn2_gamma,
                               mean2, inv_std2, grad_bn2_gamma, grad_bn2_beta, grad_bn2_in);

        // FC1 backward
        Tensor grad_h1(batch, virtual_dim_);
        for (size_t b = 0; b < batch; ++b)
            for (size_t i = 0; i < (size_t)virtual_dim_; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < (size_t)virtual_dim_; ++j)
                    sum += grad_bn2_in[b][j] * indep.w1[j][i];
                grad_h1[b][i] = sum;
            }
        for (size_t j = 0; j < (size_t)virtual_dim_; ++j)
            for (size_t i = 0; i < (size_t)virtual_dim_; ++i)
                for (size_t b = 0; b < batch; ++b)
                    indep.grad_w1[j][i] += grad_bn2_in[b][j] * indep.last_bn1_out[b][i];
        for (size_t j = 0; j < (size_t)virtual_dim_; ++j)
            for (size_t b = 0; b < batch; ++b)
                indep.grad_b1[0][j] += grad_bn2_in[b][j];

        // BN1 backward (input is shared_encoder_.last_bn2_out)
        Tensor mean1(1, virtual_dim_), inv_std1(1, virtual_dim_);
        mean1.fill(0.0);
        for (size_t f = 0; f < (size_t)virtual_dim_; ++f) {
            for (size_t b = 0; b < batch; ++b)
                mean1[0][f] += shared_encoder_.last_bn2_out[b][f];
            mean1[0][f] /= batch;
            inv_std1[0][f] = 1.0;
        }
        Tensor grad_bn1_gamma = Tensor::zeros(1, virtual_dim_);
        Tensor grad_bn1_beta = Tensor::zeros(1, virtual_dim_);
        Tensor grad_bn1_in(batch, virtual_dim_);
        batch_norm_backward_1d(shared_encoder_.last_bn2_out, grad_h1, indep.bn1_gamma,
                               mean1, inv_std1, grad_bn1_gamma, grad_bn1_beta, grad_bn1_in);

        // Accumulate for shared encoder
        for (size_t b = 0; b < batch; ++b)
            for (size_t f = 0; f < (size_t)virtual_dim_; ++f)
                grad_h_shared_accum[b][f] += grad_bn1_in[b][f];
    }

    // Shared encoder backward
    grad_h = grad_h_shared_accum;

    EncoderBlock& se = shared_encoder_;
    // BN2 backward
    const Tensor& inv_std_se = se.last_inv_std_bn2;
    const Tensor& mean_se = se.last_mean_bn2;

    Tensor grad_bn2_gamma_se = Tensor::zeros(1, virtual_dim_);
    Tensor grad_bn2_beta_se = Tensor::zeros(1, virtual_dim_);
    Tensor grad_bn2_in_se(batch, virtual_dim_);
    batch_norm_backward_1d(se.last_bn1_out, grad_h, se.bn2_gamma,
                            mean_se, inv_std_se, grad_bn2_gamma_se, grad_bn2_beta_se, grad_bn2_in_se);

    for (size_t f = 0; f < (size_t)virtual_dim_; ++f) {
        se.grad_bn2_gamma[0][f] += grad_bn2_gamma_se[0][f];
        se.grad_bn2_beta[0][f]  += grad_bn2_beta_se[0][f];
    }

    // FC2 backward
    Tensor grad_bn1_out_se(batch, virtual_dim_);
    for (size_t b = 0; b < batch; ++b)
        for (size_t i = 0; i < (size_t)virtual_dim_; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < (size_t)virtual_dim_; ++j)
                sum += grad_bn2_in_se[b][j] * se.w2[j][i];
            grad_bn1_out_se[b][i] = sum;
        }
    for (size_t j = 0; j < (size_t)virtual_dim_; ++j)
        for (size_t i = 0; i < (size_t)virtual_dim_; ++i)
            for (size_t b = 0; b < batch; ++b)
                se.grad_w2[j][i] += grad_bn2_in_se[b][j] * se.last_bn2_out[b][i];
    for (size_t j = 0; j < (size_t)virtual_dim_; ++j)
        for (size_t b = 0; b < batch; ++b)
            se.grad_b2[0][j] += grad_bn2_in_se[b][j];

    // BN1 backward (on input)
    Tensor grad_bn1_gamma_se = Tensor::zeros(1, input_dim_);
    Tensor grad_bn1_beta_se = Tensor::zeros(1, input_dim_);
    Tensor grad_bn1_in_se(batch, input_dim_);
    for (size_t b = 0; b < batch; ++b)
        for (size_t f = 0; f < (size_t)input_dim_; ++f) {
            grad_bn1_in_se[b][f] = grad_bn1_out_se[b][f] * se.bn1_gamma[0][f];
        }
    for (size_t f = 0; f < (size_t)input_dim_; ++f) {
        se.grad_bn1_gamma[0][f] = 0.0;
        se.grad_bn1_beta[0][f] = 0.0;
        for (size_t b = 0; b < batch; ++b) {
            se.grad_bn1_gamma[0][f] += grad_bn1_out_se[b][f] * (se.last_bn1_out[b][f] - se.bn1_beta[0][f]) / se.bn1_gamma[0][f];
            se.grad_bn1_beta[0][f] += grad_bn1_out_se[b][f];
        }
    }

    // FC1 backward
    for (size_t j = 0; j < (size_t)virtual_dim_; ++j)
        for (size_t i = 0; i < (size_t)input_dim_; ++i)
            for (size_t b = 0; b < batch; ++b)
                se.grad_w1[j][i] += grad_bn1_out_se[b][j] * last_h_[0][b][i];
    for (size_t j = 0; j < (size_t)virtual_dim_; ++j)
        for (size_t b = 0; b < batch; ++b)
            se.grad_b1[0][j] += grad_bn1_out_se[b][j];

    return grad_bn1_in_se;
}

void TabNet::update_weights(double learning_rate) {
    for (size_t i = 0; i < shared_encoder_.w1.rows * shared_encoder_.w1.cols; ++i)
        shared_encoder_.w1.data[i] -= learning_rate * shared_encoder_.grad_w1.data[i];
    for (size_t i = 0; i < shared_encoder_.b1.rows * shared_encoder_.b1.cols; ++i)
        shared_encoder_.b1.data[i] -= learning_rate * shared_encoder_.grad_b1.data[i];
    for (size_t i = 0; i < shared_encoder_.bn1_gamma.rows * shared_encoder_.bn1_gamma.cols; ++i)
        shared_encoder_.bn1_gamma.data[i] -= learning_rate * shared_encoder_.grad_bn1_gamma.data[i];
    for (size_t i = 0; i < shared_encoder_.bn1_beta.rows * shared_encoder_.bn1_beta.cols; ++i)
        shared_encoder_.bn1_beta.data[i] -= learning_rate * shared_encoder_.grad_bn1_beta.data[i];
    for (size_t i = 0; i < shared_encoder_.w2.rows * shared_encoder_.w2.cols; ++i)
        shared_encoder_.w2.data[i] -= learning_rate * shared_encoder_.grad_w2.data[i];
    for (size_t i = 0; i < shared_encoder_.b2.rows * shared_encoder_.b2.cols; ++i)
        shared_encoder_.b2.data[i] -= learning_rate * shared_encoder_.grad_b2.data[i];
    for (size_t i = 0; i < shared_encoder_.bn2_gamma.rows * shared_encoder_.bn2_gamma.cols; ++i)
        shared_encoder_.bn2_gamma.data[i] -= learning_rate * shared_encoder_.grad_bn2_gamma.data[i];
    for (size_t i = 0; i < shared_encoder_.bn2_beta.rows * shared_encoder_.bn2_beta.cols; ++i)
        shared_encoder_.bn2_beta.data[i] -= learning_rate * shared_encoder_.grad_bn2_beta.data[i];

    for (int s = 0; s < num_steps_; ++s) {
        IndependentBlock& indep = independent_blocks_[s];
        for (size_t i = 0; i < indep.w1.rows * indep.w1.cols; ++i)
            indep.w1.data[i] -= learning_rate * indep.grad_w1.data[i];
        for (size_t i = 0; i < indep.b1.rows * indep.b1.cols; ++i)
            indep.b1.data[i] -= learning_rate * indep.grad_b1.data[i];
        for (size_t i = 0; i < indep.bn1_gamma.rows * indep.bn1_gamma.cols; ++i)
            indep.bn1_gamma.data[i] -= learning_rate * indep.grad_bn1_gamma.data[i];
        for (size_t i = 0; i < indep.bn1_beta.rows * indep.bn1_beta.cols; ++i)
            indep.bn1_beta.data[i] -= learning_rate * indep.grad_bn1_beta.data[i];
        for (size_t i = 0; i < indep.w2.rows * indep.w2.cols; ++i)
            indep.w2.data[i] -= learning_rate * indep.grad_w2.data[i];
        for (size_t i = 0; i < indep.b2.rows * indep.b2.cols; ++i)
            indep.b2.data[i] -= learning_rate * indep.grad_b2.data[i];
        for (size_t i = 0; i < indep.bn2_gamma.rows * indep.bn2_gamma.cols; ++i)
            indep.bn2_gamma.data[i] -= learning_rate * indep.grad_bn2_gamma.data[i];
        for (size_t i = 0; i < indep.bn2_beta.rows * indep.bn2_beta.cols; ++i)
            indep.bn2_beta.data[i] -= learning_rate * indep.grad_bn2_beta.data[i];

        AttentionBlock& attn = attention_blocks_[s];
        for (size_t i = 0; i < attn.w.rows * attn.w.cols; ++i)
            attn.w.data[i] -= learning_rate * attn.grad_w.data[i];

        StepFC& fc = step_fcs_[s];
        for (size_t i = 0; i < fc.w.rows * fc.w.cols; ++i)
            fc.w.data[i] -= learning_rate * fc.grad_w.data[i];
        for (size_t i = 0; i < fc.b.rows * fc.b.cols; ++i)
            fc.b.data[i] -= learning_rate * fc.grad_b.data[i];
    }
}

std::vector<Tensor*> TabNet::parameters() { return param_list_; }
std::vector<Tensor*> TabNet::gradients() { return grad_list_; }

void TabNet::zero_grad() {
    for (Tensor* t : grad_list_) t->fill(0.0);
}

Tensor TabNet::getAttentionMask(int step) const {
    if (step < 0 || step >= num_steps_) return Tensor();
    return attention_blocks_[step].last_mask;
}

std::vector<Tensor> TabNet::getAttentionMasks() const {
    std::vector<Tensor> masks;
    for (int s = 0; s < num_steps_; ++s) {
        const Tensor& m = attention_blocks_[s].last_mask;
        if (m.rows == 0) {
            masks.push_back(Tensor(1, input_dim_));
        } else {
            Tensor avg(1, input_dim_);
            for (size_t f = 0; f < (size_t)input_dim_; ++f) {
                double sum = 0.0;
                for (size_t b = 0; b < m.rows; ++b)
                    sum += m[b][f];
                avg[0][f] = sum / m.rows;
            }
            masks.push_back(avg);
        }
    }
    return masks;
}
