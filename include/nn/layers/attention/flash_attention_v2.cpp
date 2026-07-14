#include "flash_attention_v2.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

Tensor project_right(const Tensor& input, const Tensor& weights) {
    Tensor output(input.rows, weights.cols);
    for (size_t row = 0; row < input.rows; ++row) {
        for (size_t out_feature = 0; out_feature < weights.cols; ++out_feature) {
            double value = 0.0;
            for (size_t in_feature = 0; in_feature < input.cols; ++in_feature) {
                value += input(row, in_feature) * weights(in_feature, out_feature);
            }
            output(row, out_feature) = value;
        }
    }
    return output;
}

void accumulate_weight_gradient(const Tensor& input,
                                const Tensor& grad_output,
                                Tensor& grad_weights) {
    for (size_t in_feature = 0; in_feature < input.cols; ++in_feature) {
        for (size_t out_feature = 0; out_feature < grad_output.cols; ++out_feature) {
            double value = 0.0;
            for (size_t row = 0; row < input.rows; ++row) {
                value += input(row, in_feature) * grad_output(row, out_feature);
            }
            grad_weights(in_feature, out_feature) += value;
        }
    }
}

void accumulate_input_gradient(const Tensor& grad_output,
                               const Tensor& weights,
                               Tensor& grad_input) {
    for (size_t row = 0; row < grad_output.rows; ++row) {
        for (size_t in_feature = 0; in_feature < weights.rows; ++in_feature) {
            double value = 0.0;
            for (size_t out_feature = 0; out_feature < weights.cols; ++out_feature) {
                value += grad_output(row, out_feature)
                       * weights(in_feature, out_feature);
            }
            grad_input(row, in_feature) += value;
        }
    }
}

}  // namespace

FlashAttentionV2Layer::FlashAttentionV2Layer(size_t d_model,
                                             size_t num_heads,
                                             size_t query_block_size,
                                             size_t key_block_size,
                                             bool causal)
    : d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(num_heads == 0 ? 0 : d_model / num_heads),
      query_block_size_(query_block_size),
      key_block_size_(key_block_size),
      causal_(causal),
      scale_(head_dim_ == 0 ? 0.0
                            : 1.0 / std::sqrt(static_cast<double>(head_dim_))),
      has_forward_cache_(false) {
    if (d_model_ == 0) {
        throw std::invalid_argument("FlashAttentionV2Layer: d_model must be > 0");
    }
    if (num_heads_ == 0) {
        throw std::invalid_argument("FlashAttentionV2Layer: num_heads must be > 0");
    }
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument(
            "FlashAttentionV2Layer: d_model must be divisible by num_heads");
    }
    if (query_block_size_ == 0) {
        throw std::invalid_argument(
            "FlashAttentionV2Layer: query_block_size must be > 0");
    }
    if (key_block_size_ == 0) {
        throw std::invalid_argument(
            "FlashAttentionV2Layer: key_block_size must be > 0");
    }

    W_q = Tensor::random(d_model_, d_model_, 0.01);
    W_k = Tensor::random(d_model_, d_model_, 0.01);
    W_v = Tensor::random(d_model_, d_model_, 0.01);
    W_o = Tensor::random(d_model_, d_model_, 0.01);

    grad_W_q = Tensor::zeros(d_model_, d_model_);
    grad_W_k = Tensor::zeros(d_model_, d_model_);
    grad_W_v = Tensor::zeros(d_model_, d_model_);
    grad_W_o = Tensor::zeros(d_model_, d_model_);
}

std::vector<Tensor*> FlashAttentionV2Layer::parameters() {
    return {&W_q, &W_k, &W_v, &W_o};
}

std::vector<Tensor*> FlashAttentionV2Layer::gradients() {
    return {&grad_W_q, &grad_W_k, &grad_W_v, &grad_W_o};
}

void FlashAttentionV2Layer::zero_grad() {
    grad_W_q.fill(0.0);
    grad_W_k.fill(0.0);
    grad_W_v.fill(0.0);
    grad_W_o.fill(0.0);
}

void FlashAttentionV2Layer::update_weights(double learning_rate) {
    W_q -= grad_W_q * learning_rate;
    W_k -= grad_W_k * learning_rate;
    W_v -= grad_W_v * learning_rate;
    W_o -= grad_W_o * learning_rate;
}

Tensor FlashAttentionV2Layer::forward(const Tensor& input) {
    if (input.rows != d_model_) {
        throw std::invalid_argument(
            "FlashAttentionV2Layer.forward: input rows must equal d_model");
    }
    if (input.cols == 0) {
        throw std::invalid_argument(
            "FlashAttentionV2Layer.forward: sequence length must be > 0");
    }

    const size_t seq_len = input.cols;
    Tensor token_input(seq_len, d_model_);
    for (size_t token = 0; token < seq_len; ++token) {
        for (size_t feature = 0; feature < d_model_; ++feature) {
            token_input(token, feature) = input(feature, token);
        }
    }

    last_input_ = token_input.clone();
    last_query_ = project_right(token_input, W_q);
    last_key_ = project_right(token_input, W_k);
    last_value_ = project_right(token_input, W_v);
    last_context_ = Tensor::zeros(seq_len, d_model_);
    last_logsumexp_ = Tensor(num_heads_, seq_len);
    last_logsumexp_.fill(-std::numeric_limits<double>::infinity());

    // FA-2 work partition: each Q block owns its output rows and online
    // softmax state while K/V blocks stream through the inner loop.
    for (size_t head = 0; head < num_heads_; ++head) {
        const size_t head_offset = head * head_dim_;
        for (size_t query_start = 0; query_start < seq_len;
             query_start += query_block_size_) {
            const size_t query_count =
                std::min(query_block_size_, seq_len - query_start);
            std::vector<double> row_max(
                query_count, -std::numeric_limits<double>::infinity());
            std::vector<double> row_sum(query_count, 0.0);
            Tensor output_accumulator(query_count, head_dim_);
            output_accumulator.fill(0.0);

            for (size_t key_start = 0; key_start < seq_len;
                 key_start += key_block_size_) {
                if (causal_ && key_start > query_start + query_count - 1) break;
                const size_t key_count =
                    std::min(key_block_size_, seq_len - key_start);

                for (size_t local_query = 0; local_query < query_count;
                     ++local_query) {
                    const size_t query = query_start + local_query;
                    const size_t valid_key_count = causal_
                        ? (key_start > query
                               ? 0
                               : std::min(key_count, query - key_start + 1))
                        : key_count;
                    if (valid_key_count == 0) continue;

                    std::vector<double> scores(valid_key_count, 0.0);
                    double block_max =
                        -std::numeric_limits<double>::infinity();
                    for (size_t local_key = 0; local_key < valid_key_count;
                         ++local_key) {
                        const size_t key = key_start + local_key;
                        double score = 0.0;
                        for (size_t dim = 0; dim < head_dim_; ++dim) {
                            score += last_query_(query, head_offset + dim)
                                   * last_key_(key, head_offset + dim);
                        }
                        scores[local_key] = score * scale_;
                        block_max = std::max(block_max, scores[local_key]);
                    }

                    const double old_max = row_max[local_query];
                    const double new_max = std::max(old_max, block_max);
                    const double old_scale = std::isfinite(old_max)
                        ? std::exp(old_max - new_max)
                        : 0.0;
                    for (size_t dim = 0; dim < head_dim_; ++dim) {
                        output_accumulator(local_query, dim) *= old_scale;
                    }
                    row_sum[local_query] *= old_scale;

                    for (size_t local_key = 0; local_key < valid_key_count;
                         ++local_key) {
                        const size_t key = key_start + local_key;
                        const double probability_numerator =
                            std::exp(scores[local_key] - new_max);
                        row_sum[local_query] += probability_numerator;
                        for (size_t dim = 0; dim < head_dim_; ++dim) {
                            output_accumulator(local_query, dim) +=
                                probability_numerator
                                * last_value_(key, head_offset + dim);
                        }
                    }
                    row_max[local_query] = new_max;
                }
            }

            for (size_t local_query = 0; local_query < query_count;
                 ++local_query) {
                const size_t query = query_start + local_query;
                if (!(row_sum[local_query] > 0.0)) {
                    throw std::runtime_error(
                        "FlashAttentionV2Layer.forward: empty attention row");
                }
                const double inverse_sum = 1.0 / row_sum[local_query];
                for (size_t dim = 0; dim < head_dim_; ++dim) {
                    last_context_(query, head_offset + dim) =
                        output_accumulator(local_query, dim) * inverse_sum;
                }
                last_logsumexp_(head, query) =
                    row_max[local_query] + std::log(row_sum[local_query]);
            }
        }
    }

    const Tensor token_output = project_right(last_context_, W_o);
    Tensor output(d_model_, seq_len);
    for (size_t token = 0; token < seq_len; ++token) {
        for (size_t feature = 0; feature < d_model_; ++feature) {
            output(feature, token) = token_output(token, feature);
        }
    }

    has_forward_cache_ = true;
    return output;
}

Tensor FlashAttentionV2Layer::backward(const Tensor& grad_output,
                                       double /*learning_rate*/) {
    if (!has_forward_cache_) {
        throw std::invalid_argument(
            "FlashAttentionV2Layer.backward: forward must be called first");
    }
    const size_t seq_len = last_input_.rows;
    if (grad_output.rows != d_model_ || grad_output.cols != seq_len) {
        throw std::invalid_argument(
            "FlashAttentionV2Layer.backward: grad_output shape must match the cached forward output");
    }

    Tensor token_grad_output(seq_len, d_model_);
    for (size_t token = 0; token < seq_len; ++token) {
        for (size_t feature = 0; feature < d_model_; ++feature) {
            token_grad_output(token, feature) = grad_output(feature, token);
        }
    }

    accumulate_weight_gradient(last_context_, token_grad_output, grad_W_o);
    Tensor grad_context(seq_len, d_model_);
    grad_context.fill(0.0);
    accumulate_input_gradient(token_grad_output, W_o, grad_context);

    Tensor grad_query(seq_len, d_model_);
    Tensor grad_key(seq_len, d_model_);
    Tensor grad_value(seq_len, d_model_);
    grad_query.fill(0.0);
    grad_key.fill(0.0);
    grad_value.fill(0.0);

    for (size_t head = 0; head < num_heads_; ++head) {
        const size_t head_offset = head * head_dim_;
        for (size_t query_start = 0; query_start < seq_len;
             query_start += query_block_size_) {
            const size_t query_count =
                std::min(query_block_size_, seq_len - query_start);
            std::vector<double> row_dot(query_count, 0.0);
            for (size_t local_query = 0; local_query < query_count;
                 ++local_query) {
                const size_t query = query_start + local_query;
                for (size_t dim = 0; dim < head_dim_; ++dim) {
                    row_dot[local_query] +=
                        grad_context(query, head_offset + dim)
                        * last_context_(query, head_offset + dim);
                }
            }

            for (size_t key_start = 0; key_start < seq_len;
                 key_start += key_block_size_) {
                if (causal_ && key_start > query_start + query_count - 1) break;
                const size_t key_count =
                    std::min(key_block_size_, seq_len - key_start);

                for (size_t local_query = 0; local_query < query_count;
                     ++local_query) {
                    const size_t query = query_start + local_query;
                    const size_t valid_key_count = causal_
                        ? (key_start > query
                               ? 0
                               : std::min(key_count, query - key_start + 1))
                        : key_count;
                    for (size_t local_key = 0; local_key < valid_key_count;
                         ++local_key) {
                        const size_t key = key_start + local_key;
                        double score = 0.0;
                        for (size_t dim = 0; dim < head_dim_; ++dim) {
                            score += last_query_(query, head_offset + dim)
                                   * last_key_(key, head_offset + dim);
                        }
                        score *= scale_;
                        const double probability = std::exp(
                            score - last_logsumexp_(head, query));

                        double grad_probability = 0.0;
                        for (size_t dim = 0; dim < head_dim_; ++dim) {
                            const double upstream =
                                grad_context(query, head_offset + dim);
                            grad_probability +=
                                upstream * last_value_(key, head_offset + dim);
                            grad_value(key, head_offset + dim) +=
                                probability * upstream;
                        }

                        const double grad_score = probability
                            * (grad_probability - row_dot[local_query]);
                        for (size_t dim = 0; dim < head_dim_; ++dim) {
                            grad_query(query, head_offset + dim) +=
                                scale_ * grad_score
                                * last_key_(key, head_offset + dim);
                            grad_key(key, head_offset + dim) +=
                                scale_ * grad_score
                                * last_query_(query, head_offset + dim);
                        }
                    }
                }
            }
        }
    }

    accumulate_weight_gradient(last_input_, grad_query, grad_W_q);
    accumulate_weight_gradient(last_input_, grad_key, grad_W_k);
    accumulate_weight_gradient(last_input_, grad_value, grad_W_v);

    Tensor grad_input_token(seq_len, d_model_);
    grad_input_token.fill(0.0);
    accumulate_input_gradient(grad_query, W_q, grad_input_token);
    accumulate_input_gradient(grad_key, W_k, grad_input_token);
    accumulate_input_gradient(grad_value, W_v, grad_input_token);

    Tensor grad_input(d_model_, seq_len);
    for (size_t token = 0; token < seq_len; ++token) {
        for (size_t feature = 0; feature < d_model_; ++feature) {
            grad_input(feature, token) = grad_input_token(token, feature);
        }
    }
    return grad_input;
}
