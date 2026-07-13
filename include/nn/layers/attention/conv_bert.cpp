#include "conv_bert.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace {

double stable_sigmoid(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

void append(std::vector<Tensor*>& destination, const std::vector<Tensor*>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

}  // namespace

ConvBertLayer::ConvBertLayer(size_t d_model, size_t kernel_size,
                             size_t num_heads, double alpha)
    : d_model_(d_model),
      kernel_size_(kernel_size),
      num_heads_(num_heads),
      head_dim_(num_heads == 0 ? 0 : d_model / num_heads),
      padding_(kernel_size == 0 ? 0 : (kernel_size - 1) / 2),
      attention_scale_(head_dim_ == 0 ? 0.0
                                      : 1.0 / std::sqrt(static_cast<double>(head_dim_))),
      query_projection_(d_model, d_model),
      key_projection_(d_model, d_model),
      value_projection_(d_model, d_model),
      attention_output_projection_(d_model, d_model),
      convolution_projection_(d_model, 2 * d_model),
      depthwise_weights_(d_model, kernel_size),
      depthwise_bias_(1, d_model),
      grad_depthwise_weights_(d_model, kernel_size),
      grad_depthwise_bias_(1, d_model),
      alpha_logit_(1, 1),
      grad_alpha_logit_(1, 1) {
    if (d_model == 0) {
        throw std::invalid_argument("ConvBertLayer: d_model must be positive");
    }
    if (kernel_size == 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument("ConvBertLayer: kernel_size must be positive and odd");
    }
    if (num_heads == 0 || d_model % num_heads != 0) {
        throw std::invalid_argument(
            "ConvBertLayer: num_heads must be positive and divide d_model");
    }
    set_alpha(alpha);

    const double standard_deviation = std::sqrt(2.0 / static_cast<double>(kernel_size));
    std::mt19937 generator(4242);
    std::normal_distribution<double> distribution(0.0, standard_deviation);
    for (size_t channel = 0; channel < d_model_; ++channel) {
        for (size_t offset = 0; offset < kernel_size_; ++offset) {
            depthwise_weights_(channel, offset) = distribution(generator);
        }
    }
    depthwise_bias_.fill(0.0);
    zero_grad();
}

double ConvBertLayer::alpha() const {
    return stable_sigmoid(alpha_logit_(0, 0));
}

void ConvBertLayer::set_alpha(double value) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
        throw std::invalid_argument("ConvBertLayer: alpha must be finite and in [0, 1]");
    }
    if (value == 0.0) {
        alpha_logit_(0, 0) = -36.0;
    } else if (value == 1.0) {
        alpha_logit_(0, 0) = 36.0;
    } else {
        alpha_logit_(0, 0) = std::log(value / (1.0 - value));
    }
}

Tensor ConvBertLayer::forward(const Tensor& input) {
    if (input.rows == 0) {
        throw std::invalid_argument("ConvBertLayer: input must contain at least one token");
    }
    if (input.cols != d_model_) {
        throw std::invalid_argument("ConvBertLayer: input.cols must equal d_model");
    }

    const size_t tokens = input.rows;
    last_input_ = input.clone();

    // Global multi-head self-attention branch.
    last_query_ = query_projection_.forward(input);
    last_key_ = key_projection_.forward(input);
    last_value_ = value_projection_.forward(input);
    last_attention_probs_ = Tensor(num_heads_ * tokens, tokens);
    last_attention_context_ = Tensor(tokens, d_model_);
    last_attention_context_.fill(0.0);

    for (size_t head = 0; head < num_heads_; ++head) {
        const size_t channel_begin = head * head_dim_;
        for (size_t query_token = 0; query_token < tokens; ++query_token) {
            double row_max = -INFINITY;
            for (size_t key_token = 0; key_token < tokens; ++key_token) {
                double score = 0.0;
                for (size_t channel_offset = 0; channel_offset < head_dim_;
                     ++channel_offset) {
                    const size_t channel = channel_begin + channel_offset;
                    score += last_query_(query_token, channel)
                           * last_key_(key_token, channel);
                }
                score *= attention_scale_;
                last_attention_probs_(head * tokens + query_token, key_token) = score;
                row_max = std::max(row_max, score);
            }

            double row_sum = 0.0;
            for (size_t key_token = 0; key_token < tokens; ++key_token) {
                double& probability =
                    last_attention_probs_(head * tokens + query_token, key_token);
                probability = std::exp(probability - row_max);
                row_sum += probability;
            }
            for (size_t key_token = 0; key_token < tokens; ++key_token) {
                last_attention_probs_(head * tokens + query_token, key_token) /= row_sum;
            }

            for (size_t channel_offset = 0; channel_offset < head_dim_;
                 ++channel_offset) {
                const size_t channel = channel_begin + channel_offset;
                double context = 0.0;
                for (size_t key_token = 0; key_token < tokens; ++key_token) {
                    context += last_attention_probs_(head * tokens + query_token,
                                                     key_token)
                             * last_value_(key_token, channel);
                }
                last_attention_context_(query_token, channel) = context;
            }
        }
    }
    last_attention_output_ =
        attention_output_projection_.forward(last_attention_context_);

    // Local Dense -> GLU -> depthwise Conv1D branch.
    last_convolution_pre_ = convolution_projection_.forward(input);
    last_gate_ = Tensor(tokens, d_model_);
    last_glu_ = Tensor(tokens, d_model_);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t channel = 0; channel < d_model_; ++channel) {
            const double gate = stable_sigmoid(
                last_convolution_pre_(token, d_model_ + channel));
            last_gate_(token, channel) = gate;
            last_glu_(token, channel) =
                last_convolution_pre_(token, channel) * gate;
        }
    }

    last_convolution_output_ = Tensor(tokens, d_model_);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t channel = 0; channel < d_model_; ++channel) {
            double value = depthwise_bias_(0, channel);
            for (size_t offset = 0; offset < kernel_size_; ++offset) {
                const int source = static_cast<int>(token + offset)
                                 - static_cast<int>(padding_);
                if (source >= 0 && source < static_cast<int>(tokens)) {
                    value += depthwise_weights_(channel, offset)
                           * last_glu_(static_cast<size_t>(source), channel);
                }
            }
            last_convolution_output_(token, channel) = value;
        }
    }

    Tensor output(tokens, d_model_);
    const double mix = alpha();
    for (size_t i = 0; i < output.data.size(); ++i) {
        output.data[i] = mix * last_attention_output_.data[i]
                       + (1.0 - mix) * last_convolution_output_.data[i];
    }
    return output;
}

Tensor ConvBertLayer::backward(const Tensor& grad_output, double /*learning_rate*/) {
    if (last_input_.rows == 0) {
        throw std::logic_error("ConvBertLayer: backward called before forward");
    }
    if (grad_output.rows != last_input_.rows || grad_output.cols != d_model_) {
        throw std::invalid_argument("ConvBertLayer: grad_output shape mismatch");
    }

    const size_t tokens = last_input_.rows;
    const double mix = alpha();

    Tensor grad_attention(tokens, d_model_);
    Tensor grad_convolution(tokens, d_model_);
    for (size_t i = 0; i < grad_output.data.size(); ++i) {
        grad_attention.data[i] = mix * grad_output.data[i];
        grad_convolution.data[i] = (1.0 - mix) * grad_output.data[i];
        grad_alpha_logit_(0, 0) +=
            grad_output.data[i]
            * (last_attention_output_.data[i]
               - last_convolution_output_.data[i])
            * mix * (1.0 - mix);
    }

    // Local branch: depthwise convolution, GLU, and input projection.
    Tensor grad_glu(tokens, d_model_);
    grad_glu.fill(0.0);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t channel = 0; channel < d_model_; ++channel) {
            const double upstream = grad_convolution(token, channel);
            grad_depthwise_bias_(0, channel) += upstream;
            for (size_t offset = 0; offset < kernel_size_; ++offset) {
                const int source = static_cast<int>(token + offset)
                                 - static_cast<int>(padding_);
                if (source >= 0 && source < static_cast<int>(tokens)) {
                    const size_t source_token = static_cast<size_t>(source);
                    grad_depthwise_weights_(channel, offset) +=
                        upstream * last_glu_(source_token, channel);
                    grad_glu(source_token, channel) +=
                        upstream * depthwise_weights_(channel, offset);
                }
            }
        }
    }

    Tensor grad_convolution_pre(tokens, 2 * d_model_);
    grad_convolution_pre.fill(0.0);
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t channel = 0; channel < d_model_; ++channel) {
            const double gate = last_gate_(token, channel);
            const double linear = last_convolution_pre_(token, channel);
            grad_convolution_pre(token, channel) =
                grad_glu(token, channel) * gate;
            grad_convolution_pre(token, d_model_ + channel) =
                grad_glu(token, channel) * linear * gate * (1.0 - gate);
        }
    }
    Tensor grad_input_convolution =
        convolution_projection_.backward(grad_convolution_pre, 0.0);

    // Global branch: output projection and per-head scaled dot-product attention.
    Tensor grad_context =
        attention_output_projection_.backward(grad_attention, 0.0);
    Tensor grad_query(tokens, d_model_);
    Tensor grad_key(tokens, d_model_);
    Tensor grad_value(tokens, d_model_);
    grad_query.fill(0.0);
    grad_key.fill(0.0);
    grad_value.fill(0.0);

    for (size_t head = 0; head < num_heads_; ++head) {
        const size_t channel_begin = head * head_dim_;
        Tensor grad_probabilities(tokens, tokens);
        grad_probabilities.fill(0.0);

        for (size_t query_token = 0; query_token < tokens; ++query_token) {
            for (size_t key_token = 0; key_token < tokens; ++key_token) {
                double grad_probability = 0.0;
                for (size_t channel_offset = 0; channel_offset < head_dim_;
                     ++channel_offset) {
                    const size_t channel = channel_begin + channel_offset;
                    grad_probability += grad_context(query_token, channel)
                                      * last_value_(key_token, channel);
                    grad_value(key_token, channel) +=
                        last_attention_probs_(head * tokens + query_token, key_token)
                        * grad_context(query_token, channel);
                }
                grad_probabilities(query_token, key_token) = grad_probability;
            }
        }

        for (size_t query_token = 0; query_token < tokens; ++query_token) {
            double dot = 0.0;
            for (size_t key_token = 0; key_token < tokens; ++key_token) {
                dot += last_attention_probs_(head * tokens + query_token, key_token)
                     * grad_probabilities(query_token, key_token);
            }
            for (size_t key_token = 0; key_token < tokens; ++key_token) {
                const double grad_score =
                    last_attention_probs_(head * tokens + query_token, key_token)
                    * (grad_probabilities(query_token, key_token) - dot)
                    * attention_scale_;
                for (size_t channel_offset = 0; channel_offset < head_dim_;
                     ++channel_offset) {
                    const size_t channel = channel_begin + channel_offset;
                    grad_query(query_token, channel) +=
                        grad_score * last_key_(key_token, channel);
                    grad_key(key_token, channel) +=
                        grad_score * last_query_(query_token, channel);
                }
            }
        }
    }

    Tensor grad_input_query = query_projection_.backward(grad_query, 0.0);
    Tensor grad_input_key = key_projection_.backward(grad_key, 0.0);
    Tensor grad_input_value = value_projection_.backward(grad_value, 0.0);

    Tensor grad_input(tokens, d_model_);
    for (size_t i = 0; i < grad_input.data.size(); ++i) {
        grad_input.data[i] = grad_input_convolution.data[i]
                           + grad_input_query.data[i]
                           + grad_input_key.data[i]
                           + grad_input_value.data[i];
    }
    return grad_input;
}

void ConvBertLayer::update_weights(double learning_rate) {
    query_projection_.update_weights(learning_rate);
    key_projection_.update_weights(learning_rate);
    value_projection_.update_weights(learning_rate);
    attention_output_projection_.update_weights(learning_rate);
    convolution_projection_.update_weights(learning_rate);

    for (size_t i = 0; i < depthwise_weights_.data.size(); ++i) {
        depthwise_weights_.data[i] -= learning_rate * grad_depthwise_weights_.data[i];
    }
    for (size_t i = 0; i < depthwise_bias_.data.size(); ++i) {
        depthwise_bias_.data[i] -= learning_rate * grad_depthwise_bias_.data[i];
    }
    alpha_logit_(0, 0) -= learning_rate * grad_alpha_logit_(0, 0);
}

void ConvBertLayer::zero_grad() {
    query_projection_.zero_grad();
    key_projection_.zero_grad();
    value_projection_.zero_grad();
    attention_output_projection_.zero_grad();
    convolution_projection_.zero_grad();
    grad_depthwise_weights_.fill(0.0);
    grad_depthwise_bias_.fill(0.0);
    grad_alpha_logit_.fill(0.0);
}

std::vector<Tensor*> ConvBertLayer::parameters() {
    std::vector<Tensor*> result;
    append(result, query_projection_.parameters());
    append(result, key_projection_.parameters());
    append(result, value_projection_.parameters());
    append(result, attention_output_projection_.parameters());
    append(result, convolution_projection_.parameters());
    result.push_back(&depthwise_weights_);
    result.push_back(&depthwise_bias_);
    result.push_back(&alpha_logit_);
    return result;
}

std::vector<Tensor*> ConvBertLayer::gradients() {
    std::vector<Tensor*> result;
    append(result, query_projection_.gradients());
    append(result, key_projection_.gradients());
    append(result, value_projection_.gradients());
    append(result, attention_output_projection_.gradients());
    append(result, convolution_projection_.gradients());
    result.push_back(&grad_depthwise_weights_);
    result.push_back(&grad_depthwise_bias_);
    result.push_back(&grad_alpha_logit_);
    return result;
}
