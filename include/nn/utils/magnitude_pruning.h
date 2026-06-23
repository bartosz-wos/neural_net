#ifndef MAGNITUDE_PRUNING_H
#define MAGNITUDE_PRUNING_H

#include "../core/tensor.h"
#include <vector>

// Magnitude-based weight pruning utilities — the standard mechanism for
// "Lottery Ticket" experiments (Frankle & Carlin 2019,
// https://arxiv.org/abs/1803.03635).
//
// The idea is to zero out a target fraction of weights that have the
// smallest absolute value (i.e. are "least important"). After pruning,
// the remaining non-zero weights are typically reset to their original
// initial values and the network is retrained — the lottery ticket
// hypothesis is that this pruned + reset subnetwork can match the
// performance of the full network trained from scratch.
//
// Three free functions matching the project utility style
// (cf. clip_grad_norm_.h):
//
//   size_t count_nonzero(const Tensor& t);
//       Number of entries strictly non-zero.
//
//   size_t magnitude_prune_(Tensor& w, double sparsity);
//       In-place: zero out the `sparsity` fraction of weights with
//       the smallest absolute value. Returns the number of entries
//       actually zeroed. `sparsity` must be in [0, 1]; values outside
//       that range are clamped.
//
//   size_t global_magnitude_prune_(const std::vector<Tensor*>& params, double sparsity);
//       Same idea, but the threshold is computed GLOBALLY across the
//       entire list of parameters: collect all |w| values, find the
//       (1 - sparsity)-th percentile threshold, then zero out any
//       parameter entry whose absolute value is strictly below that
//       threshold. Returns the total number of entries zeroed.
//
// All "zero out" operations are exact: the entry becomes exactly 0.0,
// not just close to it.

// Count entries strictly non-zero.
size_t count_nonzero(const Tensor& t);

// Magnitude-prune a single weight tensor (in-place).
// `sparsity` is clamped to [0, 1].
// Returns: number of entries zeroed.
size_t magnitude_prune_(Tensor& w, double sparsity);

// Magnitude-prune globally across a list of parameter tensors (in-place).
// The threshold is the (1 - sparsity)-th percentile of |w| across all params.
// `sparsity` is clamped to [0, 1]. Null tensors in the list are skipped.
// Returns: total number of entries zeroed.
size_t global_magnitude_prune_(const std::vector<Tensor*>& params, double sparsity);

#endif
