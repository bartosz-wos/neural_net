#ifndef LOSSES_H
#define LOSSES_H

#include "../core/tensor.h"
#include <string>

// Reduction modes, matching PyTorch's `reduction=` argument
// (aten/src/ATen/native/LossNLL.cpp:542-546).
#ifndef NN_REDUCTION_ENUM_DEFINED
#define NN_REDUCTION_ENUM_DEFINED
enum class Reduction { None, Mean, Sum };
#endif

// ---------------------------------------------------------------------------
// log_softmax helpers (public: callers building a custom loss need them, and
// they are the numerically-stable path CrossEntropyLoss itself takes).
// log_softmax(x)[i][j] = x[i][j] - max_j x[i][j] - log(sum_k exp(x[i][k] - max))
// The max subtraction is what keeps a +1e4 logit row from overflowing exp().
// ---------------------------------------------------------------------------
Tensor log_softmax(const Tensor& x);
// dL/dx_j = g_j - softmax(x)_j * sum_k g_k, given log_probs = log_softmax(x).
Tensor log_softmax_backward(const Tensor& grad_output, const Tensor& log_probs);

// ---------------------------------------------------------------------------
// NLLLoss — negative log-likelihood of an ALREADY-log-softmax'd input.
// loss_b = -input[b][target[b]] * w[target[b]]   (0 for ignored rows)
// mean: sum / total_weight, total_weight = B - n_ignored, or the weight sum
//       when class weights are given (LossNLL.cpp:288-292 — D1)
//
// Backward puts -w[t_b] * grad_output / total_weight at [b][t_b] and zero
// everywhere else; an ignored row's gradient row is EXACTLY zero (D2).
// ---------------------------------------------------------------------------
class NLLLoss {
public:
    NLLLoss(Reduction reduction = Reduction::Mean, int ignore_index = -100);
    void set_weight(const Tensor& weight);          // (1, C)
    Tensor forward(const Tensor& log_probs, const Tensor& targets) const;
    Tensor backward(const Tensor& log_probs, const Tensor& targets,
                    const Tensor& grad_output) const;
    double total_weight() const { return total_weight_; }
    Reduction reduction() const { return reduction_; }
    int ignore_index() const { return ignore_index_; }

private:
    void compute_total_weight(const Tensor& targets) const;
    double class_weight(size_t cls) const;

    Reduction reduction_;
    int ignore_index_;
    Tensor weight_;
    mutable double total_weight_ = 1.0;
};

// ---------------------------------------------------------------------------
// CrossEntropyLoss — softmax cross-entropy over RAW LOGITS (D4).
// Equivalent to NLLLoss(log_softmax(logits)) but folds the softmax in and
// computes the gradient in logit space directly.
// ---------------------------------------------------------------------------
class CrossEntropyLoss {
public:
    CrossEntropyLoss(Reduction reduction = Reduction::Mean, int ignore_index = -100);
    void set_weight(const Tensor& weight);
    Tensor forward(const Tensor& logits, const Tensor& targets) const;
    Tensor backward(const Tensor& logits, const Tensor& targets,
                    const Tensor& grad_output) const;
    double total_weight() const { return total_weight_; }
    Reduction reduction() const { return reduction_; }
    int ignore_index() const { return ignore_index_; }

private:
    Reduction reduction_;
    int ignore_index_;
    Tensor weight_;
    mutable double total_weight_ = 1.0;
};

// ---------------------------------------------------------------------------
// BCEWithLogitsLoss — binary cross-entropy from logits (no sigmoid needed).
// loss = (1 - target) * input - log_sigmoid(input), stably computed as
//        max(x,0) - x*target + log1p(exp(-|x|))   (Loss.cpp:349-364)
// mean divides by numel = rows*cols, NOT by rows (D7).
// ---------------------------------------------------------------------------
class BCEWithLogitsLoss {
public:
    BCEWithLogitsLoss(Reduction reduction = Reduction::Mean);
    void set_weight(const Tensor& weight);          // elementwise, shape of input
    void set_pos_weight(const Tensor& pos_weight);  // (1, 1); >= 0
    Tensor forward(const Tensor& logits, const Tensor& targets) const;
    Tensor backward(const Tensor& logits, const Tensor& targets,
                    const Tensor& grad_output) const;
    Reduction reduction() const { return reduction_; }

private:
    Reduction reduction_;
    Tensor weight_;
    Tensor pos_weight_;
};

#endif